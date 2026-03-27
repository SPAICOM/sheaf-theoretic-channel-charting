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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import lightning as l
import scipy
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from torch.utils.data import DataLoader


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

    # -------------------------------------------------------
    #     Methods for dimensionality reduction evaluation
    # -------------------------------------------------------

    @torch.no_grad()
    def build_test_trajectory(self, agent_idx: int):
        test_local_loader = DataLoader(
            self.trainer.datamodule.test_local_dataset[int(agent_idx)], batch_size=64, shuffle=False
        )
        agent = self.agents[agent_idx]

        test_embs = []
        pos = []

        for batch in test_local_loader:
            test_embs.append(agent(batch)[0])
            pos.append(batch[-1])

        embs = torch.cat(test_embs, dim=0)
        pos = torch.cat(pos, dim=0)

        embs_KDTree = KDTree(embs)
        pos_KDTree = KDTree(pos)

        return embs, pos, embs_KDTree, pos_KDTree

    def compute_continuity(
        self,
        embs: torch.tensor,
        pos: torch.tensor,
        # embs_KDTree: scipy.spatial.KDTree,
        pos_KDTree: scipy.spatial.KDTree,
        K: int,
    ) -> None:
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))

        CT = torch.zeros(N)
        for i in range(N):
            # Get K nearest neighbors in the original space (k=K+1 to exclude self)
            x = pos[i, :]
            _, Ux = pos_KDTree.query(x, k=K + 1)
            Ux = Ux[1:]  # discard self (rank 0)

            # Global ranks of all points in embedding space from i
            D_all = torch.linalg.norm(embs - embs[i, :], dim=1)
            D_all[i] = float('inf')  # exclude self
            global_ranks = torch.argsort(torch.argsort(D_all)) + 1  # 1-indexed

            # Penalty: sum max(0, rank - K) for each original K-NN
            r = global_ranks[Ux]
            CT[i] = 1 - F * torch.sum(torch.clamp(r - K, min=0))

        return torch.mean(CT)

    def compute_trustworthiness(
        self,
        embs: torch.tensor,
        pos: torch.tensor,
        embs_KDTree: scipy.spatial.KDTree,
        # pos_KDTree: scipy.spatial.KDTree,
        K: int,
    ) -> None:
        N = pos.shape[0]
        F = 2 / (K * (2 * N - 3 * K - 1))

        TW = torch.zeros(N)
        for i in range(N):
            # Get K nearest neighbors in the representation space (k=K+1 to exclude self)
            x = embs[i, :]
            _, Vx = embs_KDTree.query(x, k=K + 1)
            Vx = Vx[1:]  # discard self (rank 0)

            # Global ranks of all points in original space from i
            D_all = torch.linalg.norm(pos - pos[i, :], dim=1)
            D_all[i] = float('inf')  # exclude self
            global_ranks = torch.argsort(torch.argsort(D_all)) + 1  # 1-indexed

            # Penalty: sum max(0, rank - K) for each embedding K-NN
            r = global_ranks[Vx]
            TW[i] = 1 - F * torch.sum(torch.clamp(r - K, min=0))

        return torch.mean(TW)

    def compute_kruskal_stress(self, embs: torch.tensor, pos: torch.tensor) -> None:
        # Distance in the original space
        DD = (pos**2).sum(dim=1, keepdim=True)  # (N, 1)
        D = DD + DD.T - 2 * (pos @ pos.T)
        D = torch.clamp(D, min=0.0)
        D = torch.sqrt(D)

        # Distance in the embedding space
        DD_hat = (embs**2).sum(dim=1, keepdim=True)  # (N, 1)
        D_hat = DD_hat + DD_hat.T - 2 * (embs @ embs.T)
        D_hat = torch.clamp(D_hat, min=0.0)
        D_hat = torch.sqrt(D_hat)

        # Compute optimal scaling factor
        beta = torch.sum(D * D_hat) / torch.sum(D**2)

        # Compute Kruskal stress
        KS = torch.sqrt(torch.sum((D - beta * D_hat) ** 2) / torch.sum(D**2))

        return KS

    @abstractmethod
    @torch.no_grad()
    def _compute_FOSCTTM(self) -> None:
        pass

    def eval_all(self, K_max: int, K_min: int = 2, step: int = 1):

        # Results
        res = {'KS': [], 'CT': defaultdict(list), 'TW': defaultdict(list)}
        for agent in self.agents:
            embs, pos, embs_KDTree, pos_KDTree = self.build_test_trajectory(agent_idx=agent)
            res['KS'].append(self.compute_kruskal_stress(embs=embs, pos=pos))
            for K in range(K_min, K_max + 1, step):
                res['CT'][K].append(self.compute_continuity(embs=embs, pos=pos, pos_KDTree=pos_KDTree, K=K))
                res['TW'][K].append(self.compute_trustworthiness(embs=embs, pos=pos, embs_KDTree=embs_KDTree, K=K))
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



if __name__ == '__main__':
    pass
