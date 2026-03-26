import math

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DiagonalTransportLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim

        # Parameters of the diagonal linear function
        self.D = nn.Parameter(torch.ones(in_dim))  # Linear map

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Apply coordinate-wise scaling
        y = x * self.D

        # Return scaled mapping
        return y


class NeuralDiagSheafCC(BaseOrchestrator):
    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
        lmb_min: float = 1e-3,
        lmb_max: float = 1.0,
        lmb_schedule: str = 'cosine',
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            weight_decay=weight_decay,
            lr=lr,
        )
        self._lmb = lmb_min
        self.save_hyperparameters()

        # Network description
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Optimal transport module dictionary
        self.diagonal_layers = nn.ModuleDict(
            {
                f'{i}_{j}': nn.ModuleDict(
                    {
                        i: DiagonalTransportLayer(self.agents['0'].out_dim),
                        j: DiagonalTransportLayer(self.agents['0'].out_dim),
                    }
                )
                for (i, j) in self.hparams['edges']
            }
        )

    def on_train_epoch_start(self):
        """Update lmb according to the schedule at the start of each epoch."""
        max_epochs = self.trainer.max_epochs
        t = self.current_epoch / max(max_epochs - 1, 1)

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
            (private_outputs, total_loss) : tuple[
                                        dict[int, torch.Tensor],
                                        torch.Tensor,
                                    ]
                The tuple with the output of the network and the epoch loss.
        """
        private_outputs = self(batch)
        on_step = prefix == 'train'

        # Compute embeddings of the overlapping areas
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
        total_transport_loss = 0

        # Compute the personalized loss for each agent
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

            total_private_loss += sum(agent_losses.values())

        # Compute the transport losses
        for i, j in self.hparams['edges']:
            transport_i = self.diagonal_layers[f'{i}_{j}'][str(i)](shared_outputs[(i, j)][i])

            transport_j = self.diagonal_layers[f'{i}_{j}'][str(j)](shared_outputs[(i, j)][j])

            # Edge specific transport loss
            transport_loss = (torch.linalg.norm(transport_i - transport_j, dim=1) ** 2).mean()

            total_transport_loss += transport_loss

        total_loss = total_private_loss + self._lmb * total_transport_loss

        self.log_dict(
            {
                f'{prefix}/total_private_loss': total_private_loss,
                f'{prefix}/total_alignment_loss': total_transport_loss,
                f'{prefix}/total_loss': total_loss,
            },
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def on_train_epoch_end(self):
        pass

    def communicate(self):
        pass


if __name__ == '__main__':
    pass
