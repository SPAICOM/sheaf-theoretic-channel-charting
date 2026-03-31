"""Single-agent LightningModule.

Wraps a single Agent as a LightningModule, handling training, validation,
and test steps without an orchestrator or CombinedLoader.  The batch is a
plain tuple (xA, xP, xN, y, pos) coming from a standard DataLoader.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import scipy

import lightning as l
import matplotlib.pyplot as plt
import numpy as np
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
        n_clusters: int | None = None,
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

    # def on_train_epoch_end(self) -> None:
    #     self.plot_latent_space(
    #         split='test',
    #         prefix='trajectory_test',
    #         last_epoch_only=True,
    #         n_clusters=self.hparams.n_clusters,
    #     )
    #     self.plot_latent_space(
    #         split='train',
    #         prefix='trajectory_train',
    #         last_epoch_only=True,
    #         n_clusters=self.hparams.n_clusters,
    #     )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return {'optimizer': optimizer}

    def set_lr(self, lr: float) -> None:
        """Set a new learning rate for the optimizer.

        This method allows changing the learning rate dynamically, useful for
        fine-tuning across folds with different learning rates.

        Parameters
        ----------
        lr : float
            New learning rate to use.
        """
        self.hparams.lr = lr
        try:
            if self.trainer is not None:
                optimizers = self.trainer.optimizers
                if optimizers is not None:
                    for opt in optimizers:
                        for param_group in opt.param_groups:
                            param_group['lr'] = lr
        except RuntimeError:
            pass

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
        """Build trajectory embeddings for train or test split.

        Returns None when the requested split is empty (e.g. test_split=0).
        """
        dataset = (
            self.trainer.datamodule.train_local_dataset
            if split == 'train'
            else self.trainer.datamodule.test_local_dataset
        )
        if len(dataset[0]) == 0:
            return None

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
        chunk_size: int = 512,
    ) -> torch.Tensor:
        print('Computing Continuity')
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))

        # Batch KDTree query using all CPU cores (workers=-1)
        _, Ux_all = pos_KDTree.query(pos.cpu().numpy(), k=K + 1, workers=-1)
        Ux_all = torch.tensor(Ux_all[:, 1:], dtype=torch.long)  # (N, K), excludes self

        embs_cpu = embs.cpu()
        penalties = torch.zeros(N)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            B = end - start
            # Pairwise distances in embedding space: (B, N)
            D_chunk = torch.cdist(embs_cpu[start:end], embs_cpu)
            # Mask self
            D_chunk[torch.arange(B), torch.arange(start, end)] = float('inf')
            # Distances to the K position-space neighbors
            Ux_chunk = Ux_all[start:end]               # (B, K)
            d_neighbors = D_chunk.gather(1, Ux_chunk)  # (B, K)
            # Rank of neighbor k = 1 + #{points with smaller distance}
            # Loop over K to keep peak memory at O(B*N) instead of O(B*N*K)
            r = torch.zeros(B, K, dtype=torch.long)
            for k in range(K):
                r[:, k] = (D_chunk < d_neighbors[:, k:k+1]).sum(dim=1) + 1
            penalties[start:end] = torch.clamp(r - K, min=0).float().sum(dim=1)

        return torch.mean(1 - F * penalties)

    def compute_trustworthiness(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
        embs_KDTree: scipy.spatial.KDTree,
        K: int,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        print('Computing Trustworthiness')
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))

        # Batch KDTree query using all CPU cores (workers=-1)
        _, Vx_all = embs_KDTree.query(embs.cpu().numpy(), k=K + 1, workers=-1)
        Vx_all = torch.tensor(Vx_all[:, 1:], dtype=torch.long)  # (N, K), excludes self

        pos_cpu = pos.cpu()
        penalties = torch.zeros(N)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            B = end - start
            # Pairwise distances in position space: (B, N)
            D_chunk = torch.cdist(pos_cpu[start:end], pos_cpu)
            # Mask self
            D_chunk[torch.arange(B), torch.arange(start, end)] = float('inf')
            # Distances to the K embedding-space neighbors
            Vx_chunk = Vx_all[start:end]               # (B, K)
            d_neighbors = D_chunk.gather(1, Vx_chunk)  # (B, K)
            # Rank of neighbor k = 1 + #{points with smaller distance}
            # Loop over K to keep peak memory at O(B*N) instead of O(B*N*K)
            r = torch.zeros(B, K, dtype=torch.long)
            for k in range(K):
                r[:, k] = (D_chunk < d_neighbors[:, k:k+1]).sum(dim=1) + 1
            penalties[start:end] = torch.clamp(r - K, min=0).float().sum(dim=1)

        return torch.mean(1 - F * penalties)

    def compute_kruskal_stress(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
        M: int=1000,
        S: int=100
    ) -> torch.Tensor:
        """Compute Montecarlo-estimate of Kruskal stress metric.

        Kruskal stress measures the goodness of fit between distances
        in the original space and the embedding space using least squares.

        Parameters
        ----------
        embs : torch.Tensor
            Embeddings of shape (N, d).
        pos : torch.Tensor
            Original positions of shape (N, 2).
        M : int
            Number of sampled points 
        S : int
            Number of samples trajectories
        Returns
        -------
        torch.Tensor
            Kruskal stress value.
        """
        # Distance in the original space
        print('Computing Kruskal Stress')
        KS = torch.zeros(S)
        N = pos.shape[0]

        for i in range(S):
            idxs = torch.randint(low=0, high=N, size=(M,))

            pos_sub = pos[idxs,:]
            embs_sub = embs[idxs,:]

            DD = (pos_sub**2).sum(dim=1, keepdim=True)  # (N, 1)
            D = DD + DD.T - 2 * (pos_sub @ pos_sub.T)
            D = torch.clamp(D, min=0.0)
            D = torch.sqrt(D)

            # Distance in the embedding space
            DD_hat = (embs_sub**2).sum(dim=1, keepdim=True)  # (N, 1)
            D_hat = DD_hat + DD_hat.T - 2 * (embs_sub @ embs_sub.T)
            D_hat = torch.clamp(D_hat, min=0.0)
            D_hat = torch.sqrt(D_hat)

            # Compute optimal scaling factor
            beta = torch.sum(D * D_hat) / torch.sum(D**2)

            # Compute Kruskal stress
            KS[i] = torch.sqrt(torch.sum((D - beta * D_hat) ** 2) / torch.sum(D**2))

        return torch.mean(KS)

    def eval_all(
        self, K_max: int, K_min: int = 2, step: int = 1, split: str = 'train'
    ) -> dict[str, Any]:
        """Compute and log continuity, trustworthiness, and Kruskal stress."""
        result = self.build_trajectory(split=split)
        if result is None:
            return {}
        embs, pos, embs_KDTree, pos_KDTree = result

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
        n_clusters: int | None = None,
        split: str = 'test',
        prefix: str = 'trajectory',
        last_epoch_only: bool = False,
    ) -> None:
        """Plot latent space trajectories with optional clustering or position-based coloring.

        Creates side-by-side visualizations showing:
        - Left: Original trajectory in position space (X, Y coordinates)
        - Right: Latent space embedding (learned representation)

        Parameters
        ----------
        output_dir : Path
            Directory to save generated plots.
        n_clusters : int | None, optional
            Number of clusters for KMeans coloring. If None (default), uses position-based
            gradient coloring where x and y are normalized to [0,1] and combined as a
            weighted average. If > 0, uses KMeans clustering with 'tab10' colormap.
        split : str, optional
            Which dataset split to plot ('train', 'val', or 'test'). Default: 'test'.
        prefix : str, optional
            Filename prefix for saved plots. Default: 'trajectory'.
        last_epoch_only : bool, optional
            If True, only plots at the final epoch. Default: False.
        """
        if last_epoch_only and self.current_epoch < self.trainer.max_epochs - 1:
            return
        if self.agent.out_dim != 2:
            return

        result = self.build_trajectory(split=split)
        if result is None:
            return
        embs, pos, _, _ = result
        embs = embs.cpu()
        pos = pos.cpu()

        # Determine color scheme based on n_clusters parameter:
        # - n_clusters > 0: cluster-based coloring using KMeans
        # - n_clusters is None (default): position-gradient using normalized (x, y) coordinates
        # - n_clusters == 0: sequential index coloring (fallback)
        if n_clusters is not None and n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            color = kmeans.fit_predict(pos.numpy())
            cmap = 'tab10'
        elif n_clusters is None:
            pos_np = pos.numpy()
            x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
            y_min, y_max = pos_np[:, 1].min(), pos_np[:, 1].max()
            x_norm = (pos_np[:, 0] - x_min) / (x_max - x_min + 1e-8)
            y_norm = (pos_np[:, 1] - y_min) / (y_max - y_min + 1e-8)
            color = 0.5 * x_norm + 0.5 * y_norm
            cmap = 'viridis'
        else:
            color = None
            cmap = 'viridis'

        output_dir.mkdir(exist_ok=True, parents=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        color_val = color if color is not None else np.arange(len(pos))
        axes[0].scatter(
            pos[:, 0].numpy(), pos[:, 1].numpy(), s=10, alpha=0.7, c=color_val, cmap=cmap
        )
        axes[0].set_title('Original Trajectory')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_aspect('equal', 'box')

        axes[1].scatter(
            embs[:, 0].numpy(), embs[:, 1].numpy(), s=10, alpha=0.7, c=color_val, cmap=cmap
        )
        axes[1].set_title('Latent Space')
        axes[1].set_xlabel('Dim 1')
        axes[1].set_ylabel('Dim 2')
        axes[1].set_aspect('equal', 'box')

        fig.savefig(output_dir / f'{prefix}.png', dpi=300)
        self.logger.log_image(key=prefix, images=[fig])
        plt.close(fig)
