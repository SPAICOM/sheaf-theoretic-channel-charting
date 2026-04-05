"""Vanilla (non-federated) Channel Charting orchestrator.

This module implements the :class:`VanillaCC` orchestrator which serves as a
baseline where agents train independently without any coordination or
parameter sharing between base stations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator

if TYPE_CHECKING:
    import torch.nn as nn


class VanillaCC(BaseOrchestrator):
    """Vanilla (non-federated) Channel Charting baseline.

    This orchestrator trains each agent independently without any coordination
    or parameter sharing between base stations. Each agent optimizes only
    its own private loss function.

    This serves as a baseline to compare against federated and sheaf-based
    approaches that leverage information sharing across the wireless network.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network modules.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of neighbor indices.
        (Not used in this baseline, but required by base class.)
    lr : float
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for regularization (default: 0.0).
    lmb_min : float, optional
        Minimum value for the regularization schedule (not used, default: 1e-3).
    lmb_max : float, optional
        Maximum value for the regularization schedule (not used, default: 1.0).
    lmb_schedule : str, optional
        Schedule type for lambda (not used, default: 'cosine').

    Notes
    -----
    - This is a baseline orchestrator with no inter-agent communication.
    - Each agent trains independently on its own local data.
    - The ``neighbors`` parameter is accepted for interface compatibility
      but has no effect on training.
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
    
        # Build edge list from neighbor graph (undirected, sorted tuples)
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Shared evaluation logic for train/val/test steps.

        Computes loss for each agent independently (no coordination).

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
            - total_loss: Sum of all agent private losses
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

        total_private_loss = 0
        total_alignment_loss = 0

        # Compute loss for each agent independently
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], private_outputs[int(idx)])

            # Log individual loss components per agent
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

        # Log total private loss
        self.log(
            f'{prefix}/total_private_loss',
            total_private_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        total_loss = total_private_loss
        self.log_dict(
            {
                f'{prefix}/total_private_loss': total_private_loss,
                f'{prefix}/total_loss': total_loss,
            },
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def on_train_epoch_end(self) -> None:
        """No-op for vanilla baseline.

        Since there's no inter-agent communication or coordination,
        no action is needed at epoch end.

        Returns
        -------
        None
        """
        self.plot_latent_space(
            split='train',
            prefix='trajectory_train',
            last_epoch_only=True,
            n_clusters=self.hparams.n_clusters,
        )

    @torch.no_grad()
    def _compute_FOSCTTM(self) -> torch.Tensor:
        """Compute FOSCTTM as a baseline (no alignment applied).

        Since agents train independently, embeddings are compared directly
        without any alignment transform. This gives the reference score
        showing how misaligned uncoordinated agents are.

        Returns
        -------
        torch.Tensor
            Mean FOSCTTM score across all edges.
        """
        shared_dataset = self.trainer.datamodule.train_shared_dataset
        FOSCTTM = torch.zeros(len(self.hparams['edges']))

        for i, dataset in enumerate(shared_dataset.values()):
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)

            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            Z_i = torch.cat(embs_1, dim=0)
            Z_j = torch.cat(embs_2, dim=0)

            edge_FOSCTTM = torch.zeros(Z_j.shape[0])
            for p in range(Z_j.shape[0]):
                d = torch.linalg.norm(Z_j[p] - Z_i[p])
                Ds_1 = torch.linalg.norm(Z_j[p] - Z_i, dim=1)
                Ds_2 = torch.linalg.norm(Z_i[p] - Z_j, dim=1)
                edge_FOSCTTM[p] = 0.5 * (
                    torch.sum(Ds_1 < d) / Z_j.shape[0]
                    + torch.sum(Ds_2 < d) / Z_j.shape[0]
                )

            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
