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
...     def communicate(self, idx_i, idx_j): ...
"""

from abc import ABC, abstractmethod
from typing import Any

import lightning as l
import torch
import torch.nn as nn


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
    - Subclasses must implement: ``on_train_epoch_end``, ``_shared_eval``,
      and ``communicate``.
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
        ...     0: [xA_0, xP_0, xN_0],
        ...     1: [xA_1, xP_1, xN_1],
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

    @abstractmethod
    def communicate(
        self,
        idx_i: int,
        idx_j: int,
    ) -> torch.Tensor:
        """Perform communication between two agents.

        This method implements the communication protocol between agents.
        The specific implementation varies by orchestrator type:
        - Federated: exchange model parameters
        - Bundle/Sheaf: exchange embedding representations

        Parameters
        ----------
        idx_i : int
            Index of the first agent.
        idx_j : int
            Index of the second agent.

        Returns
        -------
        torch.Tensor
            Communication result (e.g., aggregated parameters, shared embeddings).

        Notes
        -----
        Subclasses must implement this method to define how information
        is exchanged between agents in the network.
        """
        pass


if __name__ == '__main__':
    pass
