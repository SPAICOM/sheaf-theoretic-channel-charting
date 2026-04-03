"""Diagonal Sheaf Channel Charting Orchestrator.

This orchestrator implements a diagonal sheaf approach where each edge maintains a
diagonal linear map for alignment between neighboring base stations. Unlike the full
bundle approach, this restricts transformations to diagonal matrices (element-wise
scaling), which can be solved in closed form.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DiagSheafCC(BaseOrchestrator):
    """Diagonal Sheaf Channel Charting orchestrator.

    Implements alignment via diagonal linear maps that are solved in closed form
    using a log barrier approach to ensure positive diagonal entries. Each edge
    between neighboring base stations maintains its own diagonal transformation.

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
    lmb_log_barr : float, optional
        Log barrier parameter for ensuring positive diagonal entries. Default: 1e-3.
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
        lmb_log_barr: float = 1e-3,
        transition_epoch: float | None = None,
        steepness: float = 0.1,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            weight_decay=weight_decay,
            lr=lr,
            transition_epoch=transition_epoch,
            steepness=steepness,
            n_clusters=n_clusters,
            lmb_min=lmb_min,
            lmb_max=lmb_max,
            lmb_schedule=lmb_schedule,
        )
        self._lmb = lmb_min
        self.save_hyperparameters()
        self.hparams['lmb_log_barr'] = lmb_log_barr

        # Build edge list from neighbor graph (undirected, sorted tuples)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Initialize diagonal maps as identity matrices for each edge
        d = self.agents[list(self.agents.keys())[0]].out_dim
        self.diagonal_maps: dict[tuple[str, str], torch.Tensor] = {
            edge: torch.eye(d, device=self.device) for edge in self.hparams['edges']
        }

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Update each edge's diagonal map via closed-form solution.

        Solves for the optimal diagonal transformation D that minimizes the
        alignment loss ||Z_i - Z_j @ D^T||² using a closed-form solution with
        log barrier to ensure positive diagonal entries (valid scaling factors).

        The solution is: d_i = (a_i + sqrt(a_i² + 2*lambda*b_i)) / (2*b_i)
        where a_i = Z_i · Z_j (element-wise), b_i = ||Z_j||² (element-wise)
        """
        train_shared_dataset = self.trainer.datamodule.train_shared_dataset

        diagonal_maps_temp: dict[tuple[str, str], torch.Tensor] = {}
        for dataset in train_shared_dataset.values():
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)
            edge = tuple(sorted((bs1_str, bs2_str)))

            # Accumulate embeddings over the full shared dataset
            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            Z_i = torch.cat(embs_1, dim=0)  # (N, d)
            Z_j = torch.cat(embs_2, dim=0)  # (N, d)

            # Column-wise inner products: a_i = sum over samples of Z_i[:,i] * Z_j[:,i]
            a = torch.sum(Z_i * Z_j, dim=0)  # (d,)
            # Sum of squared norms per dimension: b_i = sum over samples of Z_j[:,i]²
            b = torch.sum(Z_j * Z_j, dim=0)  # (d,)

            # Closed-form solution with log barrier to ensure positive diagonal entries
            # delta = a² + 2*λ*b ensures the argument of sqrt is positive
            delta = a**2 + 2 * self.hparams['lmb_log_barr'] * b
            d = (a + torch.sqrt(delta)) / (2 * b)

            # Construct diagonal matrix from computed scaling factors
            D = torch.diag(d)

            diagonal_maps_temp[edge] = D

        self.diagonal_maps = diagonal_maps_temp
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
        for shared data between neighboring agents using diagonal transformations.

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

        # Compute alignment losses using diagonal transformations
        # Align embeddings using element-wise scaling: ||emb_i - emb_j @ D^T||²
        for i, j in self.hparams['edges']:
            emb_i = shared_outputs[(i, j)][i]  # (B, d)
            emb_j = shared_outputs[(i, j)][j]  # (B, d)
            O_ij = self.diagonal_maps[(i, j)].to(self.device)  # (d, d)

            # Alignment loss: mean squared distance after diagonal transformation
            alignment_loss = (torch.linalg.norm(emb_i - emb_j @ O_ij.T, dim=1) ** 2).mean()

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
        in the aligned embedding space correctly matches the ground truth correspondence.

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
            edge = tuple(sorted((bs1_str, bs2_str)))

            # Accumulate embeddings over the full shared dataset
            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            # Stack all embeddings: (N, d)
            Z_i = torch.cat(embs_1, dim=0)
            Z_j = torch.cat(embs_2, dim=0)

            # Perform edge alignment using diagonal map
            Z_j_hat = Z_j @ self.diagonal_maps[edge].to(self.device)
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM: for each point, measure fraction of neighbors
            # where the nearest neighbor correctly identifies the correspondence
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i[p, :])

                # Distance from point p in j to all points in i
                Ds = torch.linalg.norm(Z_j_hat[p, :] - Z_i, dim=1)

                # Fraction of points correctly identified as closer than the true match
                edge_FOSCTTM[p] = torch.sum(Ds < d) / Z_j_hat.shape[0]

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
