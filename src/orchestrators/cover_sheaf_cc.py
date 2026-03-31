"""Cover Sheaf Channel Charting Orchestrator.

This orchestrator implements a cover-based sheaf approach where the cover of the
spatial domain is decomposed into overlapping regions, each handled by a local agent.
Alignment between neighboring base stations is computed, but reference frames are
fixed to identity (no rotation is applied).
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class CoverSheafCC(BaseOrchestrator):
    """Cover Sheaf Channel Charting orchestrator.

    Implements a cover-based approach where each base station covers an area and
    alignment is computed between overlapping regions. Unlike the full bundle
    approach, the local reference frames are fixed to identity (no rotation learning).

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network models.
    neighbors : dict[int, set[int]]
        Dictionary mapping agent indices to sets of neighboring agent indices.
    lr : float
        Learning rate for optimization.
    weight_decay : float
        L2 regularization strength.
    lmb_min : float, optional
        Minimum value for alignment loss weight. Default: 1e-3.
    lmb_max : float, optional
        Maximum value for alignment loss weight. Default: 1.0.
    lmb_schedule : str, optional
        Schedule for interpolation between lmb_min and lmb_max. Options:
        'linear', 'exponential', 'cosine'. Default: 'cosine'.
    transition_epoch : float | None, optional
        Epoch at which to transition from lmb_min to lmb_max. Default: None.
    steepness : float, optional
        Steepness of sigmoid transition. Default: 0.1.
    n_clusters : int | None, optional
        Number of clusters for latent space visualization. Default: None.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
        lmb_min: float = 1e-3,
        lmb_max: float = 1.0,
        lmb_schedule: str = 'cosine',
        transition_epoch: float | None = None,
        steepness: float = 0.1,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            n_clusters=n_clusters,
            weight_decay=weight_decay,
            lr=lr,
            transition_epoch=transition_epoch,
            steepness=steepness,
        )
        self._lmb = lmb_min

        # Build edge list from neighbor graph (undirected, sorted tuples)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Initialize local reference frames as identity matrices
        # In cover sheaf, these are NOT updated (fixed to identity)
        self.local_reference_frames: dict[str, torch.Tensor] = {
            agent: torch.eye(self.agents[agent].out_dim, device=self.device)
            for agent in self.agents
        }

    def on_train_epoch_start(self) -> None:
        """Update lmb according to the schedule at the start of each epoch.

        Computes the alignment loss weight (lmb) based on the configured schedule
        (linear, exponential, or cosine) and current epoch progress.
        """
        max_epochs = self.trainer.max_epochs
        t = self.current_epoch / max(max_epochs - 1, 1)  # normalized progress [0, 1]

        lmb_min = self.hparams['lmb_min']
        lmb_max = self.hparams['lmb_max']
        schedule = self.hparams['lmb_schedule']

        match schedule:
            case 'linear':
                self._lmb = lmb_min + (lmb_max - lmb_min) * t
            case 'exponential':
                self._lmb = lmb_min * (lmb_max / lmb_min) ** t
            case 'cosine':
                self._lmb = lmb_min + (lmb_max - lmb_min) * (1 - math.cos(math.pi * t)) / 2
            case _:
                raise ValueError(
                    f"Unknown lmb_schedule: '{schedule}'. Choose from: linear, exponential, cosine."
                )

        self.log('train/lmb', self._lmb, on_step=False, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Local reference frames fixed to identity — no alignment update.

        Unlike other bundle/sheaf approaches, the cover sheaf does not update
        reference frames during training. They remain as identity matrices.
        """
        self.plot_latent_space(
            split='train',
            prefix='trajectory_train',
            last_epoch_only=True,
            n_clusters=self.hparams.n_clusters,
        )

    def _shared_eval(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """A common step performed in the test and validation step.

        Computes private losses for each agent's local data and alignment losses
        for shared data between neighboring agents (using identity reference frames).

        Parameters
        ----------
        batch : dict[str, list[torch.Tensor]]
            The current batch containing both private and shared data.
        batch_idx : int
            The batch index.
        prefix : str
            The step type for logging purposes ('train', 'val', 'test').

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor]
            Tuple of (private outputs dict, total loss).
        """
        private_outputs = self(batch)
        on_step = prefix == 'train'

        # Compute embeddings of the overlapping areas between base stations
        shared_outputs = {
            (i, j): {
                i: self.agents[i](
                    batch[(int(i), int(j))][0],
                    triplet_mode=False,
                ),
                j: self.agents[j](
                    batch[(int(i), int(j))][1],
                    triplet_mode=False,
                ),
            }
            for (i, j) in self.hparams['edges']
        }

        total_private_loss = 0.0
        total_alignment_loss = 0.0

        # Compute the personalized loss for each agent on its private data
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], private_outputs[int(idx)])

            for loss_name, loss_val in agent_losses.items():
                self.log(
                    f'{prefix}/{loss_name}_agent_{idx}',
                    loss_val,
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=batch_size,
                    prog_bar=False,
                )

            # Alpha-weighted combination of reconstruction and triplet loss
            if 'rec_loss' in agent_losses:
                alpha = self._compute_alpha(self.current_epoch)
                for loss_name, loss_val in agent_losses.items():
                    match loss_name:
                        case 'rec_loss':  # Weight = (1 - alpha)
                            total_private_loss += (1 - alpha) * loss_val
                        case 'triplet_loss':  # Weight = alpha
                            total_private_loss += alpha * loss_val
                        case _:  # Other losses - full weight
                            total_private_loss += loss_val
            else:
                # No reconstruction loss - use full triplet loss
                total_private_loss += sum(agent_losses.values())

        # Compute alignment losses
        # Since reference frames are identity, this measures raw embedding distance
        for i, j in self.hparams['edges']:
            aligned_i = shared_outputs[(i, j)][i] @ self.local_reference_frames[i].to(self.device).T
            aligned_j = shared_outputs[(i, j)][j] @ self.local_reference_frames[j].to(self.device).T

            # Alignment loss: mean squared distance between embeddings
            alignment_loss = (torch.linalg.norm(aligned_i - aligned_j, dim=1) ** 2).mean()

            total_alignment_loss += alignment_loss

        # Total loss = private loss + lambda * alignment loss
        total_loss = total_private_loss + self._lmb * total_alignment_loss

        self.log_dict(
            {
                f'{prefix}/total_private_loss': total_private_loss,
                f'{prefix}/total_alignment_loss': total_alignment_loss,
                f'{prefix}/total_loss': total_loss,
            },
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def _compute_FOSCTTM(self, split: str = 'train') -> torch.Tensor:
        """Compute FOSCTTM (Fraction Of Successive Correct Triplet Matches) metric.

        Evaluates the quality of alignment by measuring how often the nearest neighbor
        in the embedding space correctly matches the ground truth correspondence.

        Parameters
        ----------
        split : str, optional
            Which dataset split to use ('train' or 'test'). Default: 'train'.

        Returns
        -------
        torch.Tensor
            Mean FOSCTTM score across all edges.
        """
        shared_dataset = (
            self.trainer.datamodule.train_shared_dataset
            if split == 'train'
            else self.trainer.datamodule.test_shared_dataset
        )
        FOSCTTM = torch.zeros(len(self.hparams['edges']))

        for i, dataset in enumerate(shared_dataset.values()):
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)

            # Accumulate embeddings over the full shared dataset
            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            # Stack all embeddings: (N, d)
            Z_i = torch.cat(embs_1, dim=0)
            Z_j = torch.cat(embs_2, dim=0)

            # No alignment transformation (reference frames are identity)
            Z_j_hat = Z_j
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM: for each point, measure fraction of neighbors
            # where the nearest neighbor correctly identifies the correspondence
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i[p, :])

                # Distance from point p in j to all points in i
                Ds = torch.linalg.norm(Z_j_hat[p, :] - Z_i)

                # Fraction of points correctly identified as closer than the true match
                edge_FOSCTTM[p] = torch.sum(Ds < d) / Z_j_hat.shape[0]

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
