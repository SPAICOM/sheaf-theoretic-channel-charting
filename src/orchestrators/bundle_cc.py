"""Bundle Channel Charting orchestrator with orthogonal Procrustes alignment.

This module implements the :class:`BundleCC` orchestrator which uses a
sheaf-theoretic approach with orthogonal maps to align embeddings across
base stations via the orthogonal Procrustes problem.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class BundleCC(BaseOrchestrator):
    """Bundle Channel Charting orchestrator with orthogonal alignment.

    This orchestrator uses a sheaf-theoretic approach where each edge in the
    communication graph has an associated orthogonal transformation. These
    transformations are learned via the orthogonal Procrustes problem:

        O_ij = argmin_O ||Z_i - Z_j @ O^T||_F  s.t. O^T O = I

    The solution is given by the SVD of the cross-covariance matrix Z_i^T @ Z_j.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network modules.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of neighbor indices.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for regularization (default: 0.0).
    lmb_min : float, optional
        Minimum value for the alignment loss weight (default: 1e-3).
    lmb_max : float, optional
        Maximum value for the alignment loss weight (default: 1.0).
    lmb_schedule : str, optional
        Schedule for lambda: 'linear', 'exponential', or 'cosine' (default: 'cosine').

    Attributes
    ----------
    orthogonal_maps : dict
        Dictionary mapping edge tuples to orthogonal transformation matrices.
    _lmb : float
        Current value of the alignment loss weight.

    Example
    -------
    >>> orchestrator = BundleCC(
    ...     agents={0: agent_0, 1: agent_1},
    ...     neighbors={0: {1}, 1: {0}},
    ...     lr=1e-3,
    ...     lmb_schedule='cosine',
    ... )
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
        )
        self._lmb = lmb_min

        # Build edge list from neighbor graph (undirected)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Initialize orthogonal maps as identity matrices
        # Each edge gets its own orthogonal transformation
        d = self.agents[list(self.agents.keys())[0]].out_dim
        self.orthogonal_maps = {
            edge: torch.eye(d, device=self.device) for edge in self.hparams['edges']
        }

    def on_train_epoch_start(self) -> None:
        """Update the alignment weight lambda according to the schedule.

        Computes the normalized epoch progress t in [0, 1] and updates
        ``self._lmb`` based on the configured schedule:
        - linear: lmb_min + (lmb_max - lmb_min) * t
        - exponential: lmb_min * (lmb_max / lmb_min) ^ t
        - cosine: lmb_min + (lmb_max - lmb_min) * (1 - cos(pi * t)) / 2

        Returns
        -------
        None
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
        """Update each edge's orthogonal map via orthogonal Procrustes.

        For each edge (i, j), solves the orthogonal Procrustes problem:

            O_ij = argmin_O ||Z_i - Z_j @ O^T||_F
            s.t. O^T O = I

        The solution is computed via SVD of the cross-covariance matrix:
            cross_cov = Z_i^T @ Z_j = U @ S @ V^T
            O_ij = U @ V^T

        The orthogonal maps are computed over the full shared dataset
        (all batches, not just one) for accuracy.

        Returns
        -------
        None

        Notes
        -----
        - This uses ``train_shared_dataset`` from the datamodule.
        - Embeddings are accumulated over the full dataset before SVD.
        """
        train_shared_dataset = self.trainer.datamodule.train_shared_dataset

        orthogonal_maps_temp = {}
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

            # Stack all embeddings: (N, d)
            Z_i = torch.cat(embs_1, dim=0)
            Z_j = torch.cat(embs_2, dim=0)

            # Compute cross-covariance matrix: (d, d)
            cross_cov = Z_i.T @ Z_j

            # Orthogonal Procrustes solution via SVD: O = U @ V^T
            # This gives the closest orthogonal matrix to cross_cov
            U, _, Vt = torch.linalg.svd(cross_cov)
            orthogonal_maps_temp[edge] = U @ Vt

        self.orthogonal_maps = orthogonal_maps_temp

        self.plot_latent_space(
            split='test',
            prefix='trajectory_test',
            last_epoch_only=True,
            n_clusters=self.hparams.n_clusters,
        )
        self.plot_latent_space(
            split='train',
            prefix='trajectory_train',
            last_epoch_only=True,
            n_clusters=self.hparams.n_clusters,
        )

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Shared evaluation logic for train/val/test steps.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.
        prefix : str
            Prefix for logging (e.g., 'train', 'val', 'test').

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor]
            Tuple containing:
            - outputs: Dictionary of agent embeddings
            - total_loss: Combined private + alignment loss
        """
        private_outputs = self(batch)
        on_step = prefix == 'train'

        # Compute embeddings of the overlapping areas (shared regions)
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

        total_private_loss = 0
        total_alignment_loss = 0

        # Compute private loss for each agent
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

            # Compute alpha-weighted loss if reconstruction loss exists (use_decoder=True)
            if 'rec_loss' in agent_losses:
                alpha = self._compute_alpha(self.current_epoch)
                for loss_name, loss_val in agent_losses.items():
                    match loss_name:
                        case 'rec_loss':  # Always match - weight is (1 - alpha)
                            total_private_loss += (1 - alpha) * loss_val
                        case 'triplet_loss':  # Weighted by alpha
                            total_private_loss += alpha * loss_val
                        case _:  # Other losses (e.g., alignment) - use full weight
                            total_private_loss += loss_val
            else:
                # No reconstruction loss (use_decoder=False) - use full triplet loss
                total_private_loss += sum(agent_losses.values())

        # Compute alignment loss between overlapping regions using orthogonal maps
        for i, j in self.hparams['edges']:
            emb_i = shared_outputs[(i, j)][i]  # (B, d)
            emb_j = shared_outputs[(i, j)][j]  # (B, d)
            O_ij = self.orthogonal_maps[(i, j)].to(self.device)  # (d, d)

            # Alignment loss: ||emb_i - emb_j @ O_ij^T||² per sample
            alignment_loss = (torch.linalg.norm(emb_i - emb_j @ O_ij.T, dim=1) ** 2).mean()

            total_alignment_loss += alignment_loss

        # Combined loss: private loss + lambda * alignment loss
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

    def _compute_FOSCTTM(self):
        test_shared_dataset = self.trainer.datamodule.test_shared_dataset
        FOSCTTM = torch.zeros(len(self.hparams['edges']))

        for i, dataset in enumerate(test_shared_dataset.values()):
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

            # Perform edge alignment
            Z_j_hat = Z_j @ self.orthogonal_maps[edge].to(self.device)
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i[p, :])
                Ds = torch.linalg.norm(Z_j_hat[p, :] - Z_i, dim=1)

                edge_FOSCTTM[p] = torch.sum(Ds < d) / Z_j_hat.shape[0]

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
