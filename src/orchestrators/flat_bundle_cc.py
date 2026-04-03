"""Flat Bundle Channel Charting Orchestrator.

This orchestrator implements a flat bundle approach where each agent maintains its own
local reference frame, and alignment is computed via the Kabsch algorithm on cross-covariance
terms from shared data between neighboring agents.

Unlike sheaf-based approaches, this uses a simple linear alignment without enforcing
sheaf consistency constraints across the cover.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FlatBundleCC(BaseOrchestrator):
    """Flat Bundle Channel Charting orchestrator.

    Implements alignment via Kabsch algorithm on cross-covariance terms computed
    over shared datasets between neighboring base stations. Each agent maintains
    a local reference frame that is updated each epoch to align embeddings.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network models.
    neighbors : dict[int, set[int]]
        Dictionary mapping agent indices to sets of neighboring agent indices.
    lr : float
        Learning rate for optimization.
    weight_decay : float, optional
        L2 regularization strength. Default: 0.0.
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
        weight_decay: float = 0.0,
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

        # Build edge list from neighbor graph (undirected, sorted tuples)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Initialize local reference frames as identity matrices
        # Each agent starts with its own coordinate system
        self.local_reference_frames = {
            agent: torch.eye(self.agents[agent].out_dim, device=self.device)
            for agent in self.agents
        }

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Alignment via Kabsch on cross-covariance terms computed over
        the full shared dataset (all examples, not just one batch).

        For each agent, computes cross-covariance between its embeddings and
        neighbor embeddings (aligned to neighbor's reference frame), then solves
        the Kabsch problem to find the optimal rotation for alignment.
        """

        train_shared_dataset = self.trainer.datamodule.train_shared_dataset

        # Accumulate embeddings for both agents over the full shared dataset
        shared_embeddings: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        for dataset in train_shared_dataset.values():
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)
            edge = tuple(sorted((bs1_str, bs2_str)))

            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            shared_embeddings[edge] = {
                bs1_str: torch.cat(embs_1, dim=0),  # (N, d)
                bs2_str: torch.cat(embs_2, dim=0),  # (N, d)
            }

        # Compute cross-covariance and solve alignment via Kabsch algorithm
        local_reference_frames_temp: dict[str, torch.Tensor] = {}
        for agent_str in self.agents:
            d = self.agents[agent_str].out_dim
            cross_cov = torch.zeros((d, d), device=self.device)

            for edge in self.hparams['edges']:
                if agent_str not in edge:
                    continue
                neighbor_str = edge[1] if edge[0] == agent_str else edge[0]

                Z_a = shared_embeddings[edge][agent_str]  # (N, d)
                Z_n = shared_embeddings[edge][neighbor_str]  # (N, d)
                R_n = self.local_reference_frames[neighbor_str].to(self.device)

                # Aggregate cross-covariance: R_n @ Z_n^T @ Z_a
                # Correct Procrustes target: polar_factor((Z_n @ R_n^T)^T @ Z_a)
                cross_cov += R_n @ Z_n.T @ Z_a

            # Kabsch polar factor: find closest proper rotation to cross-covariance
            # U @ Sigma @ V^T where Sigma is adjusted to ensure det(R) = +1
            U, _, Vt = torch.linalg.svd(cross_cov)
            Sigma_tilde = torch.ones(U.shape[1], device=self.device)
            Sigma_tilde[-1] = torch.linalg.det(U @ Vt)  # Ensure proper rotation
            local_reference_frames_temp[agent_str] = U @ torch.diag(Sigma_tilde) @ Vt

        # Update local reference frames for next epoch
        self.local_reference_frames = local_reference_frames_temp
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
        for shared data between neighboring agents. The total loss is a weighted
        combination of private and alignment losses.

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
            # when decoder is used (use_decoder=True)
            if 'rec_loss' in agent_losses:
                alpha = self._compute_alpha(self.current_epoch)
                for loss_name, loss_val in agent_losses.items():
                    match loss_name:
                        case 'rec_loss':  # Weight = (1 - alpha)
                            total_private_loss += (1 - alpha) * loss_val
                        case 'triplet_loss':  # Weight = alpha
                            total_private_loss += alpha * loss_val
                        case _:  # Other losses (alignment) - full weight
                            total_private_loss += loss_val
            else:
                # No reconstruction loss - use full triplet loss
                total_private_loss += sum(agent_losses.values())

        # Compute alignment losses for each edge
        # Align embeddings to each agent's local reference frame and compute
        # the squared distance between aligned embeddings from neighboring agents
        for i, j in self.hparams['edges']:
            aligned_i = shared_outputs[(i, j)][i] @ self.local_reference_frames[i].to(self.device).T
            aligned_j = shared_outputs[(i, j)][j] @ self.local_reference_frames[j].to(self.device).T

            # Edge-specific transport loss: mean squared distance between aligned embeddings
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

            # Perform edge alignment using learned reference frames
            Z_i_hat = Z_i @ self.local_reference_frames[bs1_str].to(self.device).T
            Z_j_hat = Z_j @ self.local_reference_frames[bs2_str].to(self.device).T
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM: for each point, measure fraction of neighbors
            # where the nearest neighbor correctly identifies the correspondence
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat[p, :])

                # Distance from point p in j to all points in i
                Ds_1 = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat, dim=1)

                # Distance from point p in i to all points in j
                Ds_2 = torch.linalg.norm(Z_i_hat[p, :] - Z_j_hat, dim=1)

                # Fraction of points correctly identified as closer than the true match
                edge_FOSCTTM[p] = 0.5 * (
                    torch.sum(Ds_1 < d) / Z_j_hat.shape[0] + torch.sum(Ds_2 < d) / Z_j_hat.shape[0]
                )

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
