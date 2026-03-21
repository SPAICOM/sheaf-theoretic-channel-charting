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
        lmb: float = 1.0,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            lr=lr,
        )
        self.save_hyperparameters()

        # Network description
        self.hparams['edges'] = list({
            tuple(sorted((agent, neighbor)))
            for agent in self.hparams['neighbors']
            for neighbor in self.hparams['neighbors'][agent]
        })
        
        # Optimal transport module dictionary
        self.transport_layers = nn.ModuleDict({
            f"{i}_{j}": nn.ModuleDict({
                str(i): OptimalTransportLayer(self.agents[0].out_dim),
                str(j): OptimalTransportLayer(self.agents[0].out_dim),
            })
            for (i, j) in self.hparams['edges']
        })

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
        private_outputs = self(batch)

        # Compute embeddings of the overlapping areas
        shared_outputs = {
            (i,j) : {
                i: self.agents[i](batch[(int(i),int(j))][0]),
                j: self.agents[j](batch[(int(i),int(j))][1])
            } for (i,j) in self.hparams['edges']
        }

        total_loss = 0

        # Compute the personalized loss for each agent
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            private_loss = agent.compute_loss(private_outputs[int(idx)])

            self.log(
                f'{prefix}/loss_agent_{idx}',
                loss,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )

            total_loss += private_loss

        for i, j in self.edges:
            transport_i = self.transport_layers[f"{i}_{j}"][str(i)](
                shared_outputs[(i, j)][i]
            )

            transport_j = self.transport_layers[f"{i}_{j}"][str(j)](
                shared_outputs[(i, j)][j]
            )

            transport_loss = torch.linalg.norm(transport_i - transport_j) ** 2

            total_loss += self.hparams['lmb'] * transport_loss

        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return outputs, total_loss



if __name__ == '__main__':
    pass
