import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class SheafCC(BaseOrchestrator):
    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
    ):
        super().__init__()

        # Network description
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # Optimal transport module dictionary
        self.local_reference_frames = {
            agent: torch.eye(self.agents[agent].out_dim)
            for agent in self.agents
        }

    @torch.no_grad()
    def on_train_epoch_end(
        self,
        batch: dict[str, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,self
    ):
        """Alignment step performed via Kabsch algorithm to cross-covariance terms among neighbors.
        """

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

        # Temporary buffer
        local_reference_frames_temp = {
            agent: torch.zeros((self.agents[agent].out_dim, self.agents[agent].out_dim))
            for agent in self.hparams['neighbors']
        }

        # Compute cross-covariance factors and their Kabsch polar factor
        for agent in self.hparams['neighbors']:
            cross_cov = torch.zeros((self.agents[agent].out_dim, self.agents[agent].out_dim))
            for neighbors in self.hparams['neighbors'][agent]:
                # Aggregate cross covariance
                edge = tuple(sorted((agent, str(neighbor))))
                cross_cov += (
                    (shared_outputs[edge][agent].T
                    @ (self.local_reference_frames[neighbor]
                        @ shared_outputs[edge][neighbor].T)
                    ).T
                )

            # Compute Kabsch polar factor 
            U, _, Vt = torch.linalg.svd(cross_cov)
            Sigma_tilde = torch.ones(U.shape[1])

            Sigma_tilde[-1] = torch.linalg.det(U @ Vt)
            local_reference_frames_temp[agent] = U @ torch.diag(Sigma_tilde) @ Vt

        # Update local reference frames
        self.local_reference_frames = local_reference_frames_temp

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
        private_outputs = self(batch)

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
        total_alignment_loss = 0

        # Compute the personalized loss for each agent
        for idx, agent in self.agents.items():
            batch_size = batch[int(idx)][0].size(0)
            private_loss = agent.compute_loss(private_outputs[int(idx)])

            self.log(
                f'{prefix}/loss_agent_{idx}',
                private_loss,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )

            total_private_loss += private_loss

        self.log(
            f'{prefix}/total_private_loss',
            total_private_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Compute the alignment losses
        for i, j in self.hparams['edges']:
            aligned_i = shared_outputs[(i,j)][i] @ self.local_reference_frames[i].T
            aligned_j = shared_outputs[(i,j)][j] @ self.local_reference_frames[j].T

            # Edge specific transport loss
            alignment_loss = torch.linalg.norm(aligned_i - aligned_j) ** 2

            total_alignment_loss += alignment_loss

        self.log(
            f'{prefix}/total_alignment_loss',
            total_alignment_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        total_loss = (
            total_private_loss + self.hparams['lmb'] * total_transport_loss
        )

        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def communicate(self):
        pass

if __name__ == '__main__':
    pass
