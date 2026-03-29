"""Single-agent LightningModule.

Wraps a single Agent as a LightningModule, handling training, validation,
and test steps without an orchestrator or CombinedLoader.  The batch is a
plain tuple (xA, xP, xN, y, pos) coming from a standard DataLoader.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as l
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


class SingleAgentModule(l.LightningModule):
    """LightningModule for a single base-station agent.

    Parameters
    ----------
    agent : nn.Module
        An Agent instance (Encoder + optional Decoder + SiameseLayer).
    lr : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay.
    transition_epoch : float | None
        Midpoint for the alpha sigmoid schedule (defaults to max_epochs / 2).
    steepness : float
        Steepness of the alpha sigmoid schedule.
    """

    def __init__(
        self,
        agent: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
        transition_epoch: float | None = None,
        steepness: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['agent'])
        self.agent = agent

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, ...],
        batch_idx: int,
        prefix: str,
    ) -> torch.Tensor:
        batch_size = batch[0].size(0)
        on_step = prefix == 'train'

        embeddings = self.agent(batch)
        losses = self.agent.compute_loss(batch, embeddings)

        for loss_name, loss_val in losses.items():
            self.log(
                f'{prefix}/{loss_name}',
                loss_val,
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=False,
            )

        if 'rec_loss' in losses:
            alpha = self._compute_alpha(self.current_epoch)
            total_loss = (1 - alpha) * losses['rec_loss'] + alpha * losses['triplet_loss']
        else:
            total_loss = sum(losses.values())

        self.log(
            f'{prefix}/total_loss',
            total_loss,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        return total_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, batch_idx, 'test')

    def on_train_epoch_end(self) -> None:
        self.plot_latent_space(split='test', prefix='trajectory_test', last_epoch_only=True)
        self.plot_latent_space(split='train', prefix='trajectory_train', last_epoch_only=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {'optimizer': optimizer}

    # ------------------------------------------------------------------
    # Alpha schedule (identical to BaseOrchestrator)
    # ------------------------------------------------------------------

    def _compute_alpha(self, epoch: int) -> float:
        if self.hparams.transition_epoch is None:
            transition_point = self.trainer.max_epochs / 2
        else:
            transition_point = self.hparams.transition_epoch
        steepness = self.hparams.steepness
        return 1 / (1 + math.exp(-steepness * (epoch - transition_point)))

    # ------------------------------------------------------------------
    # Evaluation utilities (adapted from BaseOrchestrator for single agent)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_trajectory(self, split: str = 'test'):
        """Build trajectory embeddings for train or test split."""
        dataset = (
            self.trainer.datamodule.train_local_dataset
            if split == 'train'
            else self.trainer.datamodule.test_local_dataset
        )
        loader = DataLoader(dataset[0], batch_size=64, shuffle=False)

        embs_list, pos_list = [], []
        for batch in loader:
            batch = [b.to(self.device) for b in batch]
            embs_list.append(self.agent(batch)[0])
            pos_list.append(batch[-1])

        embs = torch.cat(embs_list, dim=0)
        pos = torch.cat(pos_list, dim=0)

        return embs, pos, KDTree(embs.cpu()), KDTree(pos.cpu())

    def compute_continuity(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
        pos_KDTree: scipy.spatial.KDTree,
        K: int,
    ) -> torch.Tensor:
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))
        CT = torch.zeros(N)
        for i in range(N):
            _, Ux = pos_KDTree.query(pos[i, :], k=K + 1)
            Ux = Ux[1:]
            D_all = torch.linalg.norm(embs - embs[i, :], dim=1)
            D_all[i] = float('inf')
            global_ranks = torch.argsort(torch.argsort(D_all)) + 1
            CT[i] = 1 - F * torch.sum(torch.clamp(global_ranks[Ux] - K, min=0))
        return torch.mean(CT)

    def compute_trustworthiness(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
        embs_KDTree: scipy.spatial.KDTree,
        K: int,
    ) -> torch.Tensor:
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))
        TW = torch.zeros(N)
        for i in range(N):
            _, Vx = embs_KDTree.query(embs[i, :], k=K + 1)
            Vx = Vx[1:]
            D_all = torch.linalg.norm(pos - pos[i, :], dim=1)
            D_all[i] = float('inf')
            global_ranks = torch.argsort(torch.argsort(D_all)) + 1
            TW[i] = 1 - F * torch.sum(torch.clamp(global_ranks[Vx] - K, min=0))
        return torch.mean(TW)

    def compute_kruskal_stress(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        DD = (pos**2).sum(dim=1, keepdim=True)
        D = torch.sqrt(torch.clamp(DD + DD.T - 2 * (pos @ pos.T), min=0.0))
        DD_hat = (embs**2).sum(dim=1, keepdim=True)
        D_hat = torch.sqrt(torch.clamp(DD_hat + DD_hat.T - 2 * (embs @ embs.T), min=0.0))
        beta = torch.sum(D * D_hat) / torch.sum(D**2)
        return torch.sqrt(torch.sum((D - beta * D_hat) ** 2) / torch.sum(D**2))

    def eval_all(
        self,
        K_max: int,
        K_min: int = 2,
        step: int = 1,
    ) -> dict[str, Any]:
        """Compute and log continuity, trustworthiness, and Kruskal stress."""
        embs, pos, embs_KDTree, pos_KDTree = self.build_trajectory(split='test')

        res: dict[str, Any] = {
            'KS': self.compute_kruskal_stress(embs=embs, pos=pos),
            'CT': defaultdict(float),
            'TW': defaultdict(float),
        }
        for K in range(K_min, K_max + 1, step):
            res['CT'][K] = self.compute_continuity(embs=embs, pos=pos, pos_KDTree=pos_KDTree, K=K)
            res['TW'][K] = self.compute_trustworthiness(
                embs=embs, pos=pos, embs_KDTree=embs_KDTree, K=K
            )

        wandb_log = {'eval/KS': res['KS']}
        for K in range(K_min, K_max + 1, step):
            wandb_log[f'eval/CT_K{K}'] = res['CT'][K]
            wandb_log[f'eval/TW_K{K}'] = res['TW'][K]
        self.logger.experiment.log(wandb_log)

        return res

    @torch.no_grad()
    def plot_latent_space(
        self,
        output_dir: Path = Path('imgs'),
        n_clusters: int = 4,
        use_clusters: bool = True,
        split: str = 'test',
        prefix: str = 'trajectory',
        last_epoch_only: bool = False,
    ) -> None:
        if last_epoch_only and self.current_epoch < self.trainer.max_epochs - 1:
            return
        if self.agent.out_dim != 2:
            return

        embs, pos, _, _ = self.build_trajectory(split=split)
        embs = embs.cpu()
        pos = pos.cpu()

        if use_clusters and n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            color = kmeans.fit_predict(pos.numpy())
            cmap = 'tab10'
        else:
            color = None
            cmap = 'viridis'

        output_dir.mkdir(exist_ok=True, parents=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        color_val = color if color is not None else range(len(pos))
        axes[0].scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), s=10, alpha=0.7, c=color_val, cmap=cmap)
        axes[0].set_title('Original Trajectory')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_aspect('equal', 'box')

        axes[1].scatter(embs[:, 0].numpy(), embs[:, 1].numpy(), s=10, alpha=0.7, c=color_val, cmap=cmap)
        axes[1].set_title('Latent Space')
        axes[1].set_xlabel('Dim 1')
        axes[1].set_ylabel('Dim 2')
        axes[1].set_aspect('equal', 'box')

        fig.savefig(output_dir / f'{prefix}.png', dpi=300)
        self.logger.log_image(key=prefix, images=[fig])
        plt.close(fig)
