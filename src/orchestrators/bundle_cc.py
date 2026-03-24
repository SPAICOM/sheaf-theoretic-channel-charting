import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.orchestrators.base_orchestrator import BaseOrchestrator


class BundleCC(BaseOrchestrator):
    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float,
        lmb_in: float = 1e-3,
        lmb_max: float = 1.0,
        lmb_schedule: str = 'cosine',
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            weight_decay=weight_decay,
            lr=lr,
        )
        self._lmb = lmb_in

        # Network description
        self.hparams['edges'] = list(
            {
                tuple(sorted((str(agent), str(neighbor))))
                for agent in self.hparams['neighbors']
                for neighbor in self.hparams['neighbors'][agent]
            }
        )

        # One orthogonal map per edge, initialised to identity
        d = self.agents[list(self.agents.keys())[0]].out_dim
        self.orthogonal_maps = {
            edge: torch.eye(d, device=self.device)
            for edge in self.hparams['edges']
        }

    def on_train_epoch_start(self):
        """Update lmb according to the schedule at the start of each epoch."""
        max_epochs = self.trainer.max_epochs
        t = self.current_epoch / max(max_epochs - 1, 1)  # normalized progress [0, 1]

        lmb_in = self.hparams['lmb_in']
        lmb_max = self.hparams['lmb_max']
        schedule = self.hparams['lmb_schedule']

        if t == 0.0:
            self._lmb = lmb_in
        elif schedule == 'linear':
            self._lmb = lmb_in + (lmb_max - lmb_in) * t
        elif schedule == 'exponential':
            self._lmb = lmb_in * (lmb_max / lmb_in) ** t
        elif schedule == 'cosine':
            self._lmb = lmb_in + (lmb_max - lmb_in) * (1 - math.cos(math.pi * t)) / 2
        else:
            raise ValueError(f"Unknown lmb_schedule: '{schedule}'. Choose from: linear, exponential, cosine.")

        self.log('train/lmb', self._lmb, on_step=False, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def on_train_epoch_end(self):
        """Update each edge's orthogonal map by solving the orthogonal Procrustes
        problem on the full shared dataset.

        For edge (i, j), find O_ij = argmin_O ||Z_i - Z_j @ O^T||_F
        s.t. O^T O = I.  Solution via SVD of the cross-covariance Z_i^T @ Z_j.
        """
        train_shared_dataset = self.trainer.datamodule.train_shared_dataset

        orthogonal_maps_temp = {}
        for edge_key, dataset in train_shared_dataset.items():
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

            Z_i = torch.cat(embs_1, dim=0)  # (N, d)
            Z_j = torch.cat(embs_2, dim=0)  # (N, d)

            # Cross-covariance: (d, d)
            cross_cov = Z_i.T @ Z_j

            # Orthogonal Procrustes: closest proper rotation to cross_cov
            U, _, Vt = torch.linalg.svd(cross_cov)
            Sigma_tilde = torch.ones(U.shape[1], device=self.device)
            Sigma_tilde[-1] = torch.linalg.det(U @ Vt)
            orthogonal_maps_temp[edge] = U @ torch.diag(Sigma_tilde) @ Vt

        self.orthogonal_maps = orthogonal_maps_temp

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
        on_step = True if prefix == "train" else False

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

        self.log(
            f'{prefix}/total_private_loss',
            total_private_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Compute the alignment losses
        for i, j in self.hparams['edges']:
            emb_i = shared_outputs[(i, j)][i]   # (B, d)
            emb_j = shared_outputs[(i, j)][j]   # (B, d)
            O_ij = self.orthogonal_maps[(i, j)].to(self.device)  # (d, d)

            # Alignment: ||emb_i - emb_j @ O_ij^T||² per sample
            alignment_loss = (torch.linalg.norm(emb_i - emb_j @ O_ij.T, dim=1) ** 2).mean()

            total_alignment_loss += alignment_loss

        self.log(
            f'{prefix}/total_alignment_loss',
            total_alignment_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )

        total_loss = (
            total_private_loss + self._lmb * total_alignment_loss
        )

        self.log(
            f'{prefix}/total_loss',
            total_loss,
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
