import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class FlatBundleCC(BaseOrchestrator):
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
            agent: torch.eye(self.agents[agent].out_dim, device=self.device)
            for agent in self.agents
        }

    def on_train_epoch_start(self):
        """Update lmb according to the schedule at the start of each epoch."""
        max_epochs = self.trainer.max_epochs
        t = self.current_epoch / max(max_epochs - 1, 1)  # normalized progress [0, 1]

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

    @torch.no_grad()
    def on_train_epoch_end(self):
        """Alignment via Kabsch on cross-covariance terms computed over
        the full shared dataset (all examples, not just one batch).
        """

        train_shared_dataset = self.trainer.datamodule.train_shared_dataset

        # Accumulate embeddings for both agents over the full shared dataset
        shared_embeddings = {}
        for dataset in train_shared_dataset.values():
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            bs1_str = str(dataset.idx_bs_1)
            bs2_str = str(dataset.idx_bs_2)
            edge = tuple(sorted((bs1_str, bs2_str)))

            embs_1, embs_2 = [], []
            for H_1, H_2, _ in loader:
                H_1, H_2 = H_1.to(self.device), H_2.to(self.device)
                embs_1.append(self.agents[bs1_str](H_1, triplet_mode=False))
                embs_2.append(self.agents[bs2_str](H_2, triplet_mode=False))

            shared_embeddings[edge] = {
                bs1_str: torch.cat(embs_1, dim=0),  # (N, d)
                bs2_str: torch.cat(embs_2, dim=0),  # (N, d)
            }

        # Compute cross-covariance and solve alignment via Kabsch algorithm
        local_reference_frames_temp = {}
        for agent_str in self.agents:
            d = self.agents[agent_str].out_dim
            cross_cov = torch.zeros((d, d), device=self.device)

            for edge in self.hparams['edges']:
                if agent_str not in edge:
                    continue
                neighbor_str = edge[1] if edge[0] == agent_str else edge[0]

                Z_a = shared_embeddings[edge][agent_str]  # (N, d)
                Z_n = shared_embeddings[edge][neighbor_str]  # (N, d)
                R_n = self.local_reference_frames[neighbor_str].to(self.device)

                # Aggregate cross-covariance: Z_a^T @ (Z_n @ R_n^T)  → (d, d)
                cross_cov += Z_a.T @ (Z_n @ R_n.T)

            # Kabsch polar factor (closest proper rotation to cross_cov)
            U, _, Vt = torch.linalg.svd(cross_cov)
            Sigma_tilde = torch.ones(U.shape[1], device=self.device)
            Sigma_tilde[-1] = torch.linalg.det(U @ Vt)
            local_reference_frames_temp[agent_str] = U @ torch.diag(Sigma_tilde) @ Vt

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
        total_alignment_loss = 0

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

        # Compute the alignment losses
        for i, j in self.hparams['edges']:
            aligned_i = shared_outputs[(i, j)][i] @ self.local_reference_frames[i].to(self.device).T
            aligned_j = shared_outputs[(i, j)][j] @ self.local_reference_frames[j].to(self.device).T

            # Edge specific transport loss
            alignment_loss = (torch.linalg.norm(aligned_i - aligned_j, dim=1) ** 2).mean()

            total_alignment_loss += alignment_loss

        total_loss = total_private_loss + self._lmb * total_alignment_loss

        self.log_dict(
            {
                f'{prefix}/total_private_loss': total_private_loss,
                f'{prefix}/total_alignment_loss': total_alignment_loss,
                f'{prefix}/total_loss': total_loss,
            },
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )

        return private_outputs, total_loss

    def communicate(self):
        pass


if __name__ == '__main__':
    pass
