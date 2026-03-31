"""Base orchestrator class for multi-agent channel charting.

This module defines the :class:`BaseOrchestrator` abstract base class which
provides the common interface for all orchestrator implementations in the
sheaf-theoretic channel charting framework.

Orchestrators manage multiple agents (one per base station) and coordinate
their training through various aggregation strategies including:
- Federated averaging
- Bundle/Sheaf-based alignment
- Optimal transport

Example
-------
Subclasses must implement the following abstract methods:

>>> class MyOrchestrator(BaseOrchestrator):
...     def on_train_epoch_end(self): ...
...     def _shared_eval(self, batch, batch_idx, prefix): ...
...     def on_train_epoch_end(self): ...
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as l
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

# from torch.nn.parallel import parallel_apply


class BaseOrchestrator(l.LightningModule, ABC):
    """Abstract base class for multi-agent orchestrators.

    This class provides the common interface for all orchestrator implementations
    that coordinate training across multiple agents (base stations) in the
    channel charting framework.

    Parameters
    ----------
    agents : dict[int, nn.Module]
        Dictionary mapping agent indices to their neural network modules.
    neighbors : dict[int, set[int]]
        Dictionary mapping each agent index to the set of its neighbor indices.
        Defines the communication graph topology.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float, optional
        Weight decay for regularization (default: 0.0).

    Attributes
    ----------
    agents : nn.ModuleDict
        Dictionary of agent modules, stored as a ModuleDict for Lightning
        to properly track parameters.

    Raises
    ------
    AssertionError
        If the ``agents`` dictionary is empty.

    Notes
    -----
    - Subclasses must implement: ``on_train_epoch_end`` and ``_shared_eval``.
    - The ``forward`` method processes batches from all agents.
    - Logging is handled through the ``_shared_eval`` method which is called
      from ``training_step``, ``validation_step``, and ``test_step``.
    """

    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        lr: float,
        weight_decay: float = 0.0,
        transition_epoch: float | None = None,
        steepness: float = 0.1,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__()
        # Save hyperparameters but exclude agent modules (they're tracked separately)
        self.save_hyperparameters(ignore=['agents'])

        # Validate that at least one agent is provided
        assert len(agents) > 0, 'The "agents" dictionary must be not empty'

        # Store agents in a ModuleDict so Lightning can track their parameters
        # Convert integer keys to strings for compatibility with nn.ModuleDict
        self.agents = nn.ModuleDict({str(idx): agent for idx, agent in agents.items()})

    @abstractmethod
    def on_train_epoch_end(self) -> None:
        """Perform actions at the end of each training epoch.

        This method is called after each training epoch completes.
        Subclasses implement specific coordination strategies here
        (e.g., federated averaging, sheaf alignment).

        Returns
        -------
        None
        """
        pass

    def forward(
        self,
        combined_batch: dict[int, list[torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        """Forward pass through all agents.

        Processes input batches from multiple agents through their
        respective encoder networks.

        Parameters
        ----------
        combined_batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input tensors.
            Each value is a list containing [anchor, positive, negative] tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary mapping agent indices to their output embeddings.

        Example
        -------
        >>> batch = {
        ...     0: [xA_0, xP_0, xN_0, y, posA0],
        ...     1: [xA_1, xP_1, xN_1, y, posA1],
        ... }
        >>> outputs = orchestrator(batch)
        >>> # outputs[0] contains embeddings for agent 0
        """
        outputs = {}
        for idx_str, agent in self.agents.items():
            # Convert string key back to integer for batch indexing
            idx = int(idx_str)
            agent_batch = combined_batch[idx]

            # Process through agent's encoder network
            out = agent(agent_batch)
            outputs[idx] = out

        return outputs

    # def forward(
    #     self,
    #     combined_batch: dict[int, list[torch.Tensor]],
    # ) -> dict[int, torch.Tensor]:
    #     """Forward pass through all agents in parallel.

    #     Uses torch.nn.parallel.parallel_apply for GPU parallelism with
    #     automatic CPU fallback via ThreadPoolExecutor-equivalent internals.

    #     Parameters
    #     ----------
    #     combined_batch : dict[int, list[torch.Tensor]]
    #         Dictionary mapping agent indices to their input tensors.
    #         Each value is a list containing [anchor, positive, negative] tensors.

    #     Returns
    #     -------
    #     dict[int, torch.Tensor]
    #         Dictionary mapping agent indices to their output embeddings.
    #     """

    #     idx_strs = list(self.agents.keys())
    #     modules  = list(self.agents.values())
    #     inputs   = [(combined_batch[int(idx)],) for idx in idx_strs]

    #     results = parallel_apply(modules, inputs)

    #     return {int(idx): out for idx, out in zip(idx_strs, results)}

    @abstractmethod
    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Shared evaluation logic for train/val/test steps.

        This method contains the core computation logic that is reused
        across training, validation, and testing steps. It performs:
        1. Forward pass through agents
        2. Loss computation
        3. Metric logging

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.
        prefix : str
            String prefix for logging (e.g., 'train', 'val', 'test').

        Returns
        -------
        tuple[dict[int, torch.Tensor], torch.Tensor]
            Tuple containing:
            - Dictionary of agent outputs
            - Total loss for the batch

        Raises
        ------
        NotImplementedError
            This is an abstract method that subclasses must implement.
        """
        pass

    def training_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step for a single batch.

        Calls the shared evaluation logic with 'train' prefix for logging.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss tensor for backpropagation.
        """
        _, loss = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='train',
        )
        return loss

    def test_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> None:
        """Test step for a single batch.

        Calls the shared evaluation logic with 'test' prefix for logging.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        None
        """
        self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='test',
        )

    def validation_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
    ) -> dict[int, torch.Tensor]:
        """Validation step for a single batch.

        Calls the shared evaluation logic with 'val' prefix for logging.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary of agent outputs for potential downstream processing.
        """
        output, _ = self._shared_eval(
            batch=batch,
            batch_idx=batch_idx,
            prefix='val',
        )
        return output

    def predict_step(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[int, torch.Tensor]:
        """Prediction step for inference.

        Performs a forward pass without loss computation or logging.

        Parameters
        ----------
        batch : dict[int, list[torch.Tensor]]
            Dictionary mapping agent indices to their input batches.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (default: 0).

        Returns
        -------
        dict[int, torch.Tensor]
            Dictionary of agent outputs.
        """
        return self(batch)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer for training.

        Creates an Adam optimizer using the learning rate and weight decay
        specified in the hyperparameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the optimizer. Structure:
            ``{'optimizer': torch.optim.Adam}``
        """
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
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None:
            optimizers = trainer.optimizers
            if optimizers is not None:
                for opt in optimizers:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr

    def set_lmb_range(self, lmb_min: float, lmb_max: float) -> None:
        """Set new lambda range for the alignment loss weight schedule.

        Updates the lower and upper bounds used by the per-epoch lambda
        scheduler (``on_train_epoch_start``).  Also resets the current
        lambda to ``lmb_min`` so each fold starts from the bottom of
        its own range.

        Parameters
        ----------
        lmb_min : float
            New lower bound for lambda.
        lmb_max : float
            New upper bound for lambda.
        """
        if 'lmb_min' in self.hparams:
            self.hparams['lmb_min'] = lmb_min
        if 'lmb_max' in self.hparams:
            self.hparams['lmb_max'] = lmb_max
        if hasattr(self, '_lmb'):
            self._lmb = lmb_min

    def _compute_alpha(self, epoch: int) -> float:
        """Compute alpha using sigmoid function.

        Alpha controls the balance between reconstruction loss and triplet loss:
        loss = (1 - alpha) * rec_loss + alpha * triplet_loss

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        float
            Alpha value between 0 and 1.
        """
        if self.hparams.transition_epoch is None:
            transition_point = self.trainer.max_epochs / 2
        else:
            transition_point = self.hparams.transition_epoch

        steepness = self.hparams.steepness
        return 1 / (1 + math.exp(-steepness * (epoch - transition_point)))

    # -------------------------------------------------------
    #     Methods for dimensionality reduction evaluation
    # -------------------------------------------------------

    @torch.no_grad()
    def build_trajectory(self, agent_idx: int, split: str = 'train'):
        """Build trajectory embeddings for train or test split.

        Parameters
        ----------
        agent_idx : int
            Agent index
        split : str
            Dataset split, either 'train' or 'test' (default: 'test')
        """
        dataset = (
            self.trainer.datamodule.train_local_dataset
            if split == 'train'
            else self.trainer.datamodule.test_local_dataset
        )

        loader = DataLoader(dataset[agent_idx], batch_size=64, shuffle=False)
        agent = self.agents[str(agent_idx)]

        embs_list = []
        pos_list = []

        for batch in loader:
            batch = [b.to(self.device) for b in batch]
            embs_list.append(agent(batch)[0])
            pos_list.append(batch[-1])

        embs = torch.cat(embs_list, dim=0)
        pos = torch.cat(pos_list, dim=0)

        embs_KDTree = KDTree(embs.cpu())
        pos_KDTree = KDTree(pos.cpu())

        return embs, pos, embs_KDTree, pos_KDTree

    def compute_continuity(
        self,
        embs: torch.Tensor,
        pos: torch.Tensor,
        pos_KDTree: scipy.spatial.KDTree,
        K: int,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Compute continuity metric.

        Continuity measures how well the local neighborhood structure
        of the original high-dimensional space is preserved in the
        embedding space.

        Parameters
        ----------
        embs : torch.Tensor
            Embeddings of shape (N, d).
        pos : torch.Tensor
            Original positions of shape (N, 2).
        pos_KDTree : scipy.spatial.KDTree
            KDTree built from original positions.
        K : int
            Number of nearest neighbors to consider.
        chunk_size : int
            Number of rows to process at a time to bound peak memory.

        Returns
        -------
        torch.Tensor
            Mean continuity score across all points.
        """
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
            Ux_chunk = Ux_all[start:end]  # (B, K)
            d_neighbors = D_chunk.gather(1, Ux_chunk)  # (B, K)
            # Rank of neighbor k = 1 + #{points with smaller distance}
            # Loop over K to keep peak memory at O(B*N) instead of O(B*N*K)
            r = torch.zeros(B, K, dtype=torch.long)
            for k in range(K):
                r[:, k] = (D_chunk < d_neighbors[:, k : k + 1]).sum(dim=1) + 1
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
        """Compute trustworthiness metric.

        Trustworthiness measures how well the local neighborhood structure
        of the embedding space reflects the original high-dimensional space.

        Parameters
        ----------
        embs : torch.Tensor
            Embeddings of shape (N, d).
        pos : torch.Tensor
            Original positions of shape (N, 2).
        embs_KDTree : scipy.spatial.KDTree
            KDTree built from embeddings.
        K : int
            Number of nearest neighbors to consider.
        chunk_size : int
            Number of rows to process at a time to bound peak memory.

        Returns
        -------
        torch.Tensor
            Mean trustworthiness score across all points.
        """
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
            Vx_chunk = Vx_all[start:end]  # (B, K)
            d_neighbors = D_chunk.gather(1, Vx_chunk)  # (B, K)
            # Rank of neighbor k = 1 + #{points with smaller distance}
            # Loop over K to keep peak memory at O(B*N) instead of O(B*N*K)
            r = torch.zeros(B, K, dtype=torch.long)
            for k in range(K):
                r[:, k] = (D_chunk < d_neighbors[:, k : k + 1]).sum(dim=1) + 1
            penalties[start:end] = torch.clamp(r - K, min=0).float().sum(dim=1)

        return torch.mean(1 - F * penalties)

    def compute_kruskal_stress(
        self, embs: torch.Tensor, pos: torch.Tensor, M: int = 1000, S: int = 1000
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
        KS = torch.zeros(S)
        N = pos.shape[0]

        for i in range(S):
            idxs = torch.randint(
                low=0,
                high=N,
                size=(M,),
            )
            pos_sub = pos[idxs, :]
            embs_sub = embs[idxs, :]

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

    @abstractmethod
    @torch.no_grad()
    def _compute_FOSCTTM(self) -> None:
        pass

    def eval_all(
        self,
        K_max: int,
        K_min: int = 2,
        step: int = 1,
    ) -> dict[str, Any]:
        """Evaluate all metrics on the test set.

        Computes continuity, trustworthiness, Kruskal stress, and FOSCTTM
        metrics across all agents.

        Parameters
        ----------
        K_max : int
            Maximum number of nearest neighbors to consider.
        K_min : int, optional
            Minimum number of nearest neighbors to consider. Default is 2.
        step : int, optional
            Step size for iterating through K values. Default is 1.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics.
        """
        # Results
        res = {'KS': [], 'CT': defaultdict(list), 'TW': defaultdict(list)}
        for agent_idx in self.agents:
            embs, pos, embs_KDTree, pos_KDTree = self.build_trajectory(
                agent_idx=int(agent_idx),
                split='train',
            )
            res['KS'].append(self.compute_kruskal_stress(embs=embs, pos=pos))
            for K in range(K_min, K_max + 1, step):
                res['CT'][K].append(
                    self.compute_continuity(embs=embs, pos=pos, pos_KDTree=pos_KDTree, K=K)
                )
                res['TW'][K].append(
                    self.compute_trustworthiness(embs=embs, pos=pos, embs_KDTree=embs_KDTree, K=K)
                )
        res['FOSCTTM'] = self._compute_FOSCTTM()

        # Logging
        wandb_log = {}
        for K in range(K_min, K_max + 1, step):
            for agent_idx, agent in enumerate(self.agents):
                wandb_log[f'eval/CT_K{K}_agent_{agent_idx}'] = res['CT'][K][agent_idx]
                wandb_log[f'eval/TW_K{K}_agent_{agent_idx}'] = res['TW'][K][agent_idx]
        for agent_idx, agent in enumerate(self.agents):
            wandb_log[f'eval/KS_agent_{agent_idx}'] = res['KS'][agent_idx]
        wandb_log['eval/FOSCTTM'] = res['FOSCTTM']
        self.logger.experiment.log(wandb_log)

        return res

    @torch.no_grad()
    def plot_latent_space(
        self,
        output_dir: Path = Path('imgs'),
        n_clusters: int | None = None,
        split: str = 'train',
        prefix: str = 'trajectory',
        last_epoch_only: bool = False,
        sample: float = 0.35,
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
        sample : float, optional
            Fraction of data points to sample for plotting. Default: 0.35 (35%).
        """
        # Early exit: skip if not final epoch and last_epoch_only is enabled
        if last_epoch_only and self.current_epoch < self.trainer.max_epochs - 1:
            return
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]
            if agent.out_dim != 2:
                continue
            embs, pos, _, _ = self.build_trajectory(agent_idx=int(agent_idx), split=split)
            embs = embs.cpu()
            pos = pos.cpu()

            if sample < 1.0:
                n_samples = int(len(embs) * sample)
                indices = torch.randperm(len(embs))[:n_samples]
                embs = embs[indices]
                pos = pos[indices]

            # Determine color scheme based on n_clusters parameter:
            # - n_clusters > 0: cluster-based coloring using KMeans
            # - n_clusters is None (default): position-gradient using normalized (x, y) coordinates
            # - n_clusters == 0: sequential index coloring (fallback)
            if n_clusters is not None and n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(pos.numpy())
                color = cluster_labels
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

            # Left: Original trajectory (position space)
            color_val = color if color is not None else np.arange(len(pos))
            axes[0].scatter(
                pos[:, 0].numpy(),
                pos[:, 1].numpy(),
                s=10,
                alpha=0.7,
                c=color_val,
                cmap=cmap,
            )
            axes[0].set_title(f'Agent {agent_idx} Original Trajectory')
            axes[0].set_xlabel('X Position')
            axes[0].set_ylabel('Y Position')
            axes[0].set_aspect('equal', 'box')

            # Right: Latent space trajectory
            axes[1].scatter(
                embs[:, 0].numpy(),
                embs[:, 1].numpy(),
                s=10,
                alpha=0.7,
                c=color_val,
                cmap=cmap,
            )
            axes[1].set_title(f'Agent {agent_idx} Latent Space')
            axes[1].set_xlabel('Dim 1')
            axes[1].set_ylabel('Dim 2')
            axes[1].set_aspect('equal', 'box')

            # Save to file
            fig.savefig(output_dir / f'{prefix}_agent_{agent_idx}.png', dpi=300)

            # Log to wandb
            self.logger.log_image(key=f'{prefix}/agent_{agent_idx}', images=[fig])
            plt.close(fig)


if __name__ == '__main__':
    pass
