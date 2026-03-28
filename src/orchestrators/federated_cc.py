"""Federated Channel Charting orchestrator.

This module implements the :class:`FederatedCC` orchestrator which performs
federated averaging across multiple agents in a wireless network.

The key idea is to aggregate model parameters across agents that are within
communication range of each other, rather than requiring all agents to
participate in each round of training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FederatedCC(BaseOrchestrator):
    """Federated Channel Charting orchestrator.

    This orchestrator implements a neighbor-restricted Federated Averaging (FedAvg)
    strategy where each agent aggregates its model parameters with neighboring agents
    in the wireless network topology.

    For each agent ``i``, the updated parameters are computed as the average of
    the parameters of the agents in the set:
        {i} ∪ neighbors[i]

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network modules.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of neighbor indices.
        Defines which agents can communicate and share parameters.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for regularization (default: 0.0).

    Raises
    ------
    TypeError
        If agents have different model architectures.
    ValueError
        If agent parameter names, shapes, or buffers don't match.

    Notes
    -----
    - All agents must have identical architecture (validated in ``__init__``).
    - Both model parameters and buffers (e.g., BatchNorm statistics) are aggregated.
    - The aggregation is synchronous: first compute all new states, then apply them.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float = 0.0,
        transition_epoch: float | None = None,
        steepness: float = 0.1,
    ) -> None:
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            lr=lr,
            weight_decay=weight_decay,
            transition_epoch=transition_epoch,
            steepness=steepness,
        )

        self._validate_agents_for_fedavg()

    def _validate_agents_for_fedavg(self) -> None:
        """Validate that all agents have compatible architectures.

        Checks that all agents share the same:
        - Model class/type
        - Parameter names and shapes
        - Registered buffers

        Raises
        ------
        TypeError
            If agents have different model classes.
        ValueError
            If agent parameter names, shapes, or buffers don't match.

        Returns
        -------
        None
        """
        agents = list(self.agents.values())
        ref = agents[0]

        ref_params = dict(ref.named_parameters())
        ref_buffers = dict(ref.named_buffers())

        for i, agent in enumerate(agents[1:], start=1):
            if type(agent) is not type(ref):
                raise TypeError(f'Agent {i} has different class.')

            params = dict(agent.named_parameters())
            if params.keys() != ref_params.keys():
                raise ValueError(f'Agent {i} param names mismatch.')

            for k in ref_params:
                if params[k].shape != ref_params[k].shape:
                    raise ValueError(f'Shape mismatch in {k}')

            buffers = dict(agent.named_buffers())
            if buffers.keys() != ref_buffers.keys():
                raise ValueError(f'Agent {i} buffer mismatch.')

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Perform neighbor-restricted Federated Averaging.

        This method implements a localized variant of Federated Averaging
        where each agent updates its parameters by averaging with neighbors.

        For each agent ``i``, the updated parameters are computed as the
        average of the parameters of the agents in the set:
            {i} ∪ neighbors[i]

        The aggregation is performed synchronously:
        1. Compute all new averaged states and store them.
        2. Apply the updated states to all agents.

        This two-phase approach avoids bias from in-place parameter updates
        during the aggregation loop.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If an agent index is missing from ``self.hparams.neighbors``.

        Notes
        -----
        - Both model parameters and buffers are aggregated via ``state_dict()``.
        - Each agent is always included in its own aggregation set.
        - Neighborhood structure is defined in ``self.hparams.neighbors``.
        """
        # Convert ModuleDict string keys to integers for consistent indexing
        agents = {int(k): v for k, v in self.agents.items()}

        # Store new states to avoid in-place updates affecting subsequent aggregations
        new_states = {}

        for idx_i, agent_i in agents.items():
            # Include self in neighborhood for aggregation
            neigh = self.hparams.neighbors[int(idx_i)] | {int(idx_i)}

            # Initialize accumulator with zeros matching parameter shapes
            avg_state = {k: torch.zeros_like(v) for k, v in agent_i.state_dict().items()}

            # Accumulate parameters from all neighbors in the aggregation set
            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()

                for k in avg_state:
                    avg_state[k] += state_j[k]

            # Normalize by number of agents in the aggregation set
            for k in avg_state:
                avg_state[k] = avg_state[k].float() / len(neigh)

            # Store result (defer loading until all computations complete)
            new_states[idx_i] = avg_state

        # Apply all updates synchronously after computation completes
        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])

        self.plot_latent_space(split='test', prefix='trajectory_test')
        self.plot_latent_space(split='train', prefix='trajectory_train')

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Shared evaluation logic for train/val/test steps.

        Computes forward pass and loss for all agents, then logs metrics.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
            Each batch contains [anchor, positive, negative] tensors.
        batch_idx : int
            Index of the current batch.
        prefix : str
            Prefix for logging (e.g., 'train', 'val', 'test').

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor]
            Tuple containing:
            - outputs: Dictionary of agent embeddings
            - total_loss: Sum of all agent losses
        """
        outputs = self(batch)
        total_loss = 0

        # Enable step-level logging only during training
        on_step = prefix == 'train'

        # Compute loss for each agent independently
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], outputs[int(idx)])

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
                loss = 0.0
                for loss_name, loss_val in agent_losses.items():
                    match loss_name:
                        case 'rec_loss':  # Always match - weight is (1 - alpha)
                            loss += (1 - alpha) * loss_val
                        case 'triplet_loss':  # Weighted by alpha
                            loss += alpha * loss_val
                        case _:  # Other losses (e.g., alignment) - use full weight
                            loss += loss_val
            else:
                # No reconstruction loss (use_decoder=False) - use full triplet loss
                loss = sum(agent_losses.values())
            total_loss += loss

        # Log total loss for monitoring
        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return outputs, total_loss


if __name__ == '__main__':
    pass
