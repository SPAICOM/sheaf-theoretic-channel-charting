from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """
    Abstract base class for agents in a multi-agent learning framework.

    This class defines the common interface and shared structure that
    all agents must follow.
    Each agent is modeled as a PyTorch module and is responsible for:
        - Producing an embedding or output via a forward pass
        - Computing its own training loss
        - Managing communication with neighboring agents

    The class is designed to be used within a coordinated training setup
    (e.g., an orchestrator), where multiple agents interact and learn jointly.

    Parameters
    ----------
    idx : int
        Unique identifier of the agent.
    emb_dim : int
        Dimensionality of the agent's embedding space.

    Attributes
    ----------
    idx : int
        Unique identifier of the agent.
    emb_dim : int
        Dimension of the embedding produced by the agent.
    neighbors : set[int]
        Set of indices corresponding to neighboring agents with which
        this agent can communicate.
    """

    def __init__(
        self,
        idx: int,
        emb_dim: int,
    ) -> None:
        """
        Initialize the base agent.

        Parameters
        ----------
        idx : int
            Unique identifier assigned to the agent.
        emb_dim : int
            Embedding dimension for the agent outputs.
        """
        super().__init__()

        self.idx: int = idx
        self.emb_dim: int = emb_dim
        self.neighbors: set[int] = set()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass.

        Defines how the agent maps an input tensor to an output embedding
        or prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor (typically an embedding of dimension `emb_dim`).
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the agent-specific loss.

        Each agent defines its own objective function, enabling heterogeneous
        learning strategies within the same system.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor used for loss computation.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        pass

    def add_neighbor(
        self,
        idx: int,
    ) -> None:
        """
        Add a single neighboring agent.

        Parameters
        ----------
        idx : int
            Identifier of the neighboring agent.
        """
        self.neighbors.add(idx)

        return None

    def add_neighbors(
        self,
        neighbors: set[int],
    ) -> None:
        """
        Add multiple neighboring agents.

        Parameters
        ----------
        neighbors : set[int]
            Set of agent indices to be added as neighbors.
        """
        self.neighbors.update(neighbors)

        return None

    def communicate(
        self,
        x: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        """
        Perform communication with a neighboring agent.

        This default implementation enforces that communication can only occur
        with registered neighbors. Subclasses can override this method to
        define specific communication protocols.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor or message to be communicated.
        idx : int
            Identifier of the target agent.

        Returns
        -------
        torch.Tensor
            Output tensor after communication.

        Raises
        ------
        AssertionError
            If the target agent is not a registered neighbor.
        """
        assert idx in self.neighbors, (
            f'Agent {idx} is not a neighbor of agent {self.idx}'
        )

        # Default behavior: identity (no-op communication)
        return x
