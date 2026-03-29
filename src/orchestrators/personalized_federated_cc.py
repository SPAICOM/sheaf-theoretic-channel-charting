"""Personalized Federated Channel Charting orchestrator.

This module implements the :class:`PersonalizedFederatedCC` orchestrator which
performs partial federated averaging, where only a subset of parameters
(e.g., late encoder/decoder layers) are shared across agents while
early layers remain personalized.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class PersonalizedFederatedCC(BaseOrchestrator):
    """Semi-personalized Federated Channel Charting orchestrator.

    This orchestrator implements a hybrid approach where:
    - Shared parameters (e.g., late encoder/decoder layers) are aggregated
      across neighboring agents via federated averaging.
    - Private parameters (e.g., early encoder layers) remain local to each agent.

    This allows agents to benefit from collaborative learning on high-level
    features while retaining personalized representations for low-level features.

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
    encoder_split : float, optional
        Fraction of encoder layers to keep private (default: 0.5).
        Higher values = more personalized early layers.
    decoder_split : float, optional
        Fraction of decoder layers to keep private (default: 0.5).
        Higher values = more personalized late layers.

    Attributes
    ----------
    shared_keys : set
        Parameter keys that are shared/aggregated across agents.
    private_keys : set
        Parameter keys that remain local to each agent.

    Example
    -------
    >>> orchestrator = PersonalizedFederatedCC(
    ...     agents={0: agent_0, 1: agent_1},
    ...     neighbors={0: {1}, 1: {0}},
    ...     lr=1e-3,
    ...     encoder_split=0.3,  # First 30% of encoder layers stay private
    ... )
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float = 0.0,
        encoder_split: float = 0.5,
        decoder_split: float = 0.5,
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

        self.encoder_split = encoder_split
        self.decoder_split = decoder_split

        self._validate_agents_for_fedavg()
        self._identify_shared_keys()

    def _identify_shared_keys(self) -> None:
        """Identify which parameter keys to share vs keep private.

        Determines the split between shared and private parameters based on
        ``encoder_split`` and ``decoder_split`` fractions:
        - Early encoder/decoder layers → private (personalized)
        - Late encoder/decoder layers → shared (aggregated)

        Returns
        -------
        None
        """
        ref_agent = next(iter(self.agents.values()))
        state_keys = list(ref_agent.state_dict().keys())

        self.shared_keys: set[str] = set()
        self.private_keys: set[str] = set()

        # Collect encoder and decoder layer indices
        encoder_layers: list[int] = []
        decoder_layers: list[int] = []

        for k in state_keys:
            if k.startswith('encoder.linear_layers.') or k.startswith('encoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    encoder_layers.append(int(parts[2]))
            elif k.startswith('decoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    decoder_layers.append(int(parts[2]))

        encoder_layers = sorted(encoder_layers)
        decoder_layers = sorted(decoder_layers)

        # Compute split indices based on fractions
        enc_split_idx = int(len(encoder_layers) * self.encoder_split)
        dec_split_idx = int(len(decoder_layers) * self.decoder_split)

        # Layers after split index are shared; before are private
        shared_encoder_layers = set(encoder_layers[enc_split_idx:])
        shared_decoder_layers = set(decoder_layers[dec_split_idx:])

        # Classify each parameter key as shared or private
        for k in state_keys:
            # Handle encoder parameters
            if k.startswith('encoder.linear_layers.') or k.startswith('encoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_id = int(parts[2])
                    if layer_id in shared_encoder_layers:
                        self.shared_keys.add(k)
                    else:
                        self.private_keys.add(k)
                else:
                    self.private_keys.add(k)
            # Handle decoder parameters
            elif k.startswith('decoder.layers.'):
                parts = k.split('.')
                if len(parts) > 2 and parts[2].isdigit():
                    layer_id = int(parts[2])
                    if layer_id in shared_decoder_layers:
                        self.shared_keys.add(k)
                    else:
                        self.private_keys.add(k)
                else:
                    self.private_keys.add(k)
            # All other parameters default to private
            else:
                self.private_keys.add(k)

    def _validate_agents_for_fedavg(self) -> None:
        """Validate that all agents have compatible architectures.

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
        """Perform neighbor-restricted FedAvg on shared parameters only.

        Aggregates only the shared parameters (defined by ``shared_keys``) across
        each agent and its neighbors. Private parameters remain unchanged.

        The aggregation set for agent ``i`` is: {i} ∪ neighbors[i]

        Returns
        -------
        None
        """
        # Convert ModuleDict string keys to integers
        agents = {int(k): v for k, v in self.agents.items()}
        new_states = {}

        for idx_i, agent_i in agents.items():
            # Include self in aggregation set
            neigh = self.hparams.neighbors[int(idx_i)] | {int(idx_i)}
            state_i = agent_i.state_dict()

            # Initialize accumulator for shared parameters only
            avg_state = {k: torch.zeros_like(state_i[k]) for k in self.shared_keys}

            # Aggregate shared parameters from all neighbors
            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()
                for k in self.shared_keys:
                    avg_state[k] += state_j[k]

            # Normalize by number of agents in aggregation set
            for k in avg_state:
                avg_state[k] = avg_state[k].float() / len(neigh)

            # Merge aggregated shared params with private params
            new_state = {}
            for k in state_i:
                if k in self.shared_keys:
                    new_state[k] = avg_state[k]
                else:
                    new_state[k] = state_i[k]

            new_states[idx_i] = new_state

        # Apply all updates synchronously
        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])

        self.plot_latent_space(split='test', prefix='trajectory_test', last_epoch_only=True)
        self.plot_latent_space(split='train', prefix='trajectory_train', last_epoch_only=True)

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
            - total_loss: Sum of all agent losses
        """
        outputs = self(batch)
        total_loss = 0

        on_step = prefix == 'train'

        # Compute loss for each agent
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], outputs[int(idx)])

            # Log individual loss components
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

        # Log total loss
        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return outputs, total_loss

    @torch.no_grad()
    def _compute_FOSCTTM(self) -> None:
        # TODO: Implement FOSCTTM computation for personalized federated CC
        pass


if __name__ == '__main__':
    pass
