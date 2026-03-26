"""Vanilla (non-federated) Channel Charting orchestrator.

This module implements the :class:`VanillaCC` orchestrator which serves as a
baseline where agents train independently without any coordination or
parameter sharing between base stations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


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
    ) -> None:
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            weight_decay=weight_decay,
            lr=lr,
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

        total_private_loss = 0

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

            total_private_loss += sum(agent_losses.values())

        # Log total private loss
        self.log(
            f'{prefix}/total_private_loss',
            total_private_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )

        # No alignment loss in vanilla baseline - total loss is just private loss
        total_loss = total_private_loss

        return private_outputs, total_loss

    def on_train_epoch_end(self) -> None:
        """No-op for vanilla baseline.

        Since there's no inter-agent communication or coordination,
        no action is needed at epoch end.

        Returns
        -------
        None
        """
        pass

    def communicate(
        self,
        idx_i: int,
        idx_j: int,
    ) -> torch.Tensor:
        """Communication between two agents (no-op for vanilla baseline).

        Parameters
        ----------
        idx_i : int
            Index of the first agent.
        idx_j : int
            Index of the second agent.

        Returns
        -------
        torch.Tensor
            Empty tensor (placeholder for interface compatibility).
        """
        return torch.tensor([])


if __name__ == '__main__':
    pass
