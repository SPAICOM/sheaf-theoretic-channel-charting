import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class PersonalizedFederatedCC(BaseOrchestrator):
    """
    Semi-personalized federated orchestrator.

    Aggregates only a subset of parameters (e.g., late encoder/decoder layers)
    while leaving early layers personalized at each agent.
    """

    def __init__(
        self,
        agents: dict[int, torch.nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
        encoder_split: float = 0.5,
        decoder_split: float = 0.5,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.encoder_split = encoder_split
        self.decoder_split = decoder_split

        self._validate_agents_for_fedavg()
        self._identify_shared_keys()

    def _identify_shared_keys(self):
        """Determine which keys to aggregate vs keep local.
        """
        ref_agent = next(iter(self.agents.values()))
        state_keys = list(ref_agent.state_dict().keys())

        self.shared_keys = set()
        self.private_keys = set()

        # Encoder layer indices
        encoder_layers = set()
        decoder_layers = set()
        for k in state_keys:
            if k.startswith("encoder.linear_layers.") or k.startswith("encoder.layers."):
                parts = k.split(".")
                if parts[2].isdigit():
                    encoder_layers.add(int(parts[2]))
            elif k.startswith("decoder.layers."):
                parts = k.split(".")
                if parts[2].isdigit():
                    decoder_layers.add(int(parts[2]))

        encoder_layers = sorted(encoder_layers)
        decoder_layers = sorted(decoder_layers)

        # Split indices
        enc_split_idx = int(len(encoder_layers) * self.encoder_split)
        dec_split_idx = int(len(decoder_layers) * self.decoder_split)

        shared_encoder_layers = set(encoder_layers[enc_split_idx:])
        shared_decoder_layers = set(decoder_layers[dec_split_idx:])

        # Assign keys
        for k in state_keys:
            # Encoder
            if k.startswith("encoder.linear_layers.") or k.startswith("encoder.layers."):
                parts = k.split(".")
                layer_id = int(parts[2])
                if layer_id in shared_encoder_layers:
                    self.shared_keys.add(k)
                else:
                    self.private_keys.add(k)
            # Decoder
            elif k.startswith("decoder.layers."):
                parts = k.split(".")
                layer_id = int(parts[2])
                if layer_id in shared_decoder_layers:
                    self.shared_keys.add(k)
                else:
                    self.private_keys.add(k)
            else:
                # Other parameters are private by default
                self.private_keys.add(k)

    @torch.no_grad()
    def on_train_epoch_end(self):
        """
        Neighbor-restricted FedAvg on shared parameters only.
        Early layers remain personalized.
        """
        agents = {int(k): v for k, v in self.agents.items()}
        new_states = {}

        for idx_i, agent_i in agents.items():
            neigh = self.hparams.neighbors[idx_i] | {idx_i}
            state_i = agent_i.state_dict()

            # Initialize shared part accumulator
            avg_state = {k: torch.zeros_like(v) for k in self.shared_keys}

            # Aggregate neighbors
            for idx_j in neigh:
                state_j = agents[idx_j].state_dict()
                for k in self.shared_keys:
                    avg_state[k] += state_j[k]

            # Average
            for k in avg_state:
                avg_state[k] = avg_state[k].float() / len(neigh)

            # Merge with private
            new_state = {}
            for k in state_i:
                if k in self.shared_keys:
                    new_state[k] = avg_state[k]
                else:
                    new_state[k] = state_i[k]

            new_states[idx_i] = new_state

        # Apply updates synchronously
        for idx_i, agent in agents.items():
            agent.load_state_dict(new_states[idx_i])


    def _validate_agents_for_fedavg(self):
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

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : dict[int, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (outputs, total_loss) : tuple[
                                        dict[int, torch.Tensor],
                                        torch.Tensor,
                                    ]
                The tuple with the output of the network and the epoch loss.
        """
        outputs = self(batch)
        total_loss = 0

        on_step = True if prefix == "train" else False

        # Compute the personalized loss for each agent
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            agent_losses = agent.compute_loss(batch[int(idx)], outputs[int(idx)])

            for loss_name, loss_val in agent_losses.items():
                self.log(
                    f'{prefix}/{loss_name}_agent_{idx}',
                    loss_val,
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=batch_size,
                    prog_bar=False,
                )

            loss = sum(agent_losses.values())
            total_loss += loss

        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return outputs, total_loss

    def communicate(self, idx_i: int, idx_j: int):
        pass


if __name__ == '__main__':
    pass
