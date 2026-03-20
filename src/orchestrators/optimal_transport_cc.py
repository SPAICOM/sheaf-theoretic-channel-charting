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
        agents: list[nn.Module],
        lr: float,
        L: torch.Tensor,
        B: torch.Tensor,
        n: int,
        lmb: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Agents list
        self.hparams['agents'] = nn.ModuleList(agents)

        # TODO
        # Network description
        self.hparams['edges'] = None

    def on_train_epoch_end(self):
        dataloader = self.trainer.datamodule.train_dataloader()

        self.eval()

        with torch.no_grad():
            # Reset local aggregators of cross-covariance
            for agent in self.hparams.agents:
                agent.reset_epoch_statistics()

            # Compute and aggregate embeddings
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)

                output = self(batch)

                for i, j in self.hparams.edges:
                    self.hparams.agents[i].accumulate_statistics(
                        output[i][0], output[j][0], self.hparams.agents[j].R
                    )
                    self.hparams.agents[j].accumulate_statistics(
                        output[j][0], output[i][0], self.hparams.agents[i].R
                    )

            # Perform reference alignment
            for agent in self.hparams.agents:
                agent.update_reference_frame()

        self.train()

        return None

    def _shared_eval(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : dict[str, list[torch.Tensor]]
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

        loss = 0

        # Compute the personalized loss for each agent
        for idx, agent in self.hparams.agents.items():
            loss += agent.compute_loss(outputs[idx])

        # Get the alignment loss by making neirby agents communicate
        for agent, neighbors in self.hparams.neighbors.items():
            for neighbor in neighbors:
                self.communicate(agent, neighbor)
        pass

    def _shared_eval(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """A common step performed in the test and validation step.

        Args:
            batch : dict[str, list[torch.Tensor]]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (output, total_loss) : tuple[dict[int, torch.Tensor], torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        R = torch.zeros_like(self.hparams.L)
        main_loss = 0
        reg_loss = 0
        output = self(batch)
        E = torch.cat(
            [output[agent.idx] for agent in self.hparams.agents], dim=0
        )

        for agent in self.hparams.agents:
            main_loss += agent.compute_loss(output[agent.idx])
            R[
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
                agent.idx * self.hparams.n : (agent.idx + 1) * self.hparams.n,
            ] = agent.R

        REP = R @ E

        # Node mask
        B = len(self.hparams.agents)
        T = E.shape[1]

        node_vals = E.view(B, self.hparams.n, T)
        node_mask = node_vals.abs().sum(dim=1) != 0

        # Edge mask
        # Need it to zero-out edges where at least one
        # Base-station does not observe the trajectory

        edge_i = torch.tensor(
            [e[0] for e in self.hparams.edges], device=E.device
        )
        edge_j = torch.tensor(
            [e[1] for e in self.hparams.edges], device=E.device
        )

        edge_mask = (node_mask[edge_i] & node_mask[edge_j]).float()

        for i in range(self.hparams.n):
            reg_loss += self.hparams.lmb * torch.sum(
                (edge_mask * self.hparams.B.T @ REP[i :: self.hparams.n, :])
                ** 2
            )

        total_loss = main_loss + reg_loss

        self.log(
            f'{prefix}/main_loss_epoch',
            main_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f'{prefix}/reg_loss_epoch', reg_loss, on_step=False, on_epoch=True
        )
        self.log(
            f'{prefix}/total_loss_epoch',
            total_loss,
            on_step=False,
            on_epoch=True,
        )

        return output, total_loss


if __name__ == '__main__':
    pass
