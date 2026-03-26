import math

import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class OptimalTransportLayer(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim

        # Parameters of the affine function
        self.M = nn.Parameter(torch.eye(in_dim))  # Linear map
        self.b = nn.Parameter(torch.zeros(in_dim))  # Bias
        self.a = nn.Parameter(torch.zeros(in_dim))  # Log of the scaling vector

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure a > 0
        a = torch.exp(self.a)

        # Apply affine map
        y = torch.matmul(x, self.M.T) - self.b

        # Return scaled mapping
        return y / a


class OptimalTransportCC(BaseOrchestrator):
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
        self.transport_layers = nn.ModuleDict(
            {
                f'{i}_{j}': nn.ModuleDict(
                    {
                        i: OptimalTransportLayer(self.agents['0'].out_dim),
                        j: OptimalTransportLayer(self.agents['0'].out_dim),
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
            transport_i = self.transport_layers[f'{i}_{j}'][str(i)](shared_outputs[(i, j)][i])

            transport_j = self.transport_layers[f'{i}_{j}'][str(j)](shared_outputs[(i, j)][j])

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
    def _compute_FOSCTTM(
        self
    ):
        test_shared_dataset = self.trainer.datamodule.test_shared_dataset
        FOSCTTM = torch.zeros(len(self.hparams['edges']))

        for i, dataset in enumerate(train_shared_dataset.values()):
        
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
            Z_i_hat = self.transport_layers[f'{i}_{j}'][str(i)](Z_i)
            Z_i_hat = self.transport_layers[f'{i}_{j}'][str(i)](Z_i)
            edge_FOSCTTM = torch.zeros(Z_j_hat.shape[0])

            # Point-wise FOSCTTM
            for p in range(Z_j_hat.shape[0]):
                d = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat[p, :])

                Ds_1 = torch.linalg.norm(Z_j_hat[p, :] - Z_i_hat)
                Ds_2 = torch.linalg.norm(Z_i_hat[p, :] - Z_j_hat)

                edge_FOSCTTM[p] = 0.5 * (
                    torch.sum(Ds_1 < d) / Z_j_hat.shape[0] + 
                    torch.sum(Ds_2 < d) / Z_j_hat.shape[0]
                )
                
            FOSCTTM[i] = torch.mean(edge_FOSCTTM)

        return torch.mean(FOSCTTM)


if __name__ == '__main__':
    pass
