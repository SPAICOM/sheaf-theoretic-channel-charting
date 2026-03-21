"""
Neural network architectures used for Siamese learning on CSI data.

The implementations are inspired by the paper:
"Siamese Neural Networks for Wireless Positioning and Channel Charting".

This module defines:
    - Encoder: compresses CSI features into a low-dimensional embedding
    - Decoder: reconstructs CSI from embeddings (optional autoencoder mode)
    - SiameseNN: module implementing Siamese / triplet learning
"""

import torch
import torch.nn as nn

from src.agents.utils import SiameseLayer


class Encoder(nn.Module):
    """
    Encoder network that compresses high-dimensional CSI features
    into a low-dimensional embedding.

    The architecture progressively halves the feature dimension
    at each hidden layer until reaching the desired output dimension.

    Example:
        in_dim = 256, num_hidden_layers = 3

        256 -> 128 -> 64 -> out_dim

    Parameters
    ----------
    in_dim : int
        Dimension of the input feature vector.
    out_dim : int, default=2
        Dimension of the output embedding (e.g., 2D channel chart).
    num_hidden_layers : int, default=3
        Number of linear layers in the encoder.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        # Activation function used between hidden layers
        self.act = nn.ReLU()

        # Container for linear layers
        self.linear_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        dim = in_dim

        # Build hidden layers that progressively reduce dimensionality
        for _ in range(self.num_hidden_layers - 1):
            self.linear_layers.append(nn.Linear(dim, dim // 2))
            self.norm_layers.append(nn.LayerNorm(dim // 2))
            dim = dim // 2

        # Final projection layer producing the embedding
        self.linear_layers.append(nn.Linear(dim, self.out_dim))
        # self.norm_layers.append(nn.BatchNorm1d(self.out_dim))

    def forward(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Encoded embedding of shape (batch_size, out_dim).
        """
        # Apply activation to all layers except the last
        for i in range(self.num_hidden_layers - 1):
            x = self.act(self.linear_layers[i](x))
            x = self.norm_layers[i](x)

        # Final linear projection (no activation)
        return self.linear_layers[-1](x)


class Decoder(nn.Module):
    """
    Decoder network used when the model operates in autoencoder mode.

    The decoder reconstructs the original CSI features from
    the low-dimensional embedding produced by the encoder.

    The architecture mirrors the encoder by progressively
    increasing the feature dimension.

    Example:
        in_dim = 2, num_hidden_layers = 6

        2 -> 4 -> 8 -> 16 -> ... -> out_dim

    Parameters
    ----------
    in_dim : int, default=2
        Dimension of the embedding space.
    out_dim : int, default=256
        Dimension of the reconstructed feature vector.
    num_hidden_layers : int, default=6
        Number of linear layers in the decoder.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 256,
        num_hidden_layers: int = 6,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        # Activation function
        self.act = nn.ReLU()

        # Container for decoder layers
        self.layers = nn.ModuleList()

        dim = in_dim

        # Hidden layers progressively expand dimensionality
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(dim, dim * 2))
            dim = dim * 2

        # Final reconstruction layer
        self.layers.append(nn.Linear(dim, self.out_dim))

    def forward(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            Embedding tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed feature tensor of shape (batch_size, out_dim).
        """
        # Apply activation to all layers except the last
        for i in range(self.num_hidden_layers - 1):
            x = self.act(self.layers[i](x))

        # Final linear reconstruction
        return self.layers[-1](x)


class Agent(nn.Module):
    """
    PyTorch module implementing a Siamese neural network.

    This model learns embeddings such that:
    - Similar CSI samples are mapped close together
    - Dissimilar samples are mapped farther apart

    The network supports:
        - Contrastive loss
        - Triplet loss
        - Optional autoencoder reconstruction

    Architecture
    ------------
    Input CSI -> Encoder -> Embedding
                         -> Decoder (optional)

    The SiameseLayer computes the loss based on the embeddings.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    distance_mode : str
        Distance metric used in the Siamese loss ('euclidean' or 'cosine').
    loss_mode : str
        Loss type ('contrastive' or 'triplet').
    out_dim : int, default=2
        Output embedding dimension.
    num_hidden_layers : int, default=6
        Number of hidden layers in encoder/decoder.
    autoenc : bool, default=False
        Whether to include a decoder and reconstruct inputs.
    margin : float, default=1.0
        Margin parameter used in Siamese losses.
    lr : float, default=1e-3
        Learning rate for the optimizer.
    """

    def __init__(
        self,
        in_dim: int,
        distance_mode: str,
        loss_mode: str,
        out_dim: int = 2,
        num_hidden_layers: int = 6,
        autoenc: bool = False,
        margin: float = 1.0,
        lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate configuration
        assert distance_mode in {'euclidean', 'cosine'}, (
            'Provide a valid distance mode'
        )
        assert loss_mode in {'contrastive', 'triplet'}, (
            'Provide a valid layer mode'
        )

        self.autoenc = autoenc
        self.distance_mode = distance_mode
        self.loss_mode = loss_mode

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.margin = margin
        self.lr = lr

        # ----------------------------------------------------------
        # Architecture selection
        # ----------------------------------------------------------

        if autoenc:
            # In autoencoder mode the model reconstructs the input
            assert in_dim == out_dim, (
                'Unsolvable design choices! Provide a combination for in'
                'and out dimension coherent with the chosen architecture'
            )

            # Encoder compresses input to 2D embedding
            self.encoder = Encoder(
                in_dim=in_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            )

            # Decoder reconstructs the original feature space
            self.decoder = Decoder(
                in_dim=out_dim,
                out_dim=in_dim,
                num_hidden_layers=num_hidden_layers,
            )

            # Siamese loss layer
            self.siamese = SiameseLayer(
                loss_mode=loss_mode,
                distance_mode=distance_mode,
                margin=margin,
            )

        else:
            # Standard Siamese embedding network
            self.encoder = Encoder(
                in_dim=in_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
            )

            # No reconstruction
            self.decoder = nn.Identity()

            self.siamese = SiameseLayer(
                loss_mode=loss_mode,
                distance_mode=distance_mode,
                margin=margin,
            )

    def forward(
        self,
        batch,
        triplet_mode: bool = True,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Forward pass used by training, validation, and test steps.

        Parameters
        ----------
        batch : tuple
            Batch containing:
            - xA : torch.Tensor
                Anchor samples.
            - xP : torch.Tensor
                Positive samples.
            - xN : torch.Tensor | None
                Negative samples (None when using contrastive loss).
            - y : torch.Tensor
                Target tensor used by the Siamese loss.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
            embA : torch.Tensor
                Embedding of anchor samples.

            embP : torch.Tensor
                Embedding of positive samples.

            embN : torch.Tensor | None
                Embedding of negative samples (None in contrastive mode).

            y : torch.Tensor
                Target tensor used to compute the loss.
        """

        if triplet_mode:
            xA, xP, xN, y = batch

            # Compute embeddings
            embA = self.decoder(self.encoder(xA))
            embP = self.decoder(self.encoder(xP))

            # Negative sample may be None in contrastive mode
            embN = self.decoder(self.encoder(xN)) if xN is not None else None

            out = (embA, embP, embN, y)

        else:
            out = self.decoder(self.encoder(batch))
        return out

    def compute_loss(
        self,
        batch,
    ) -> torch.Tensor:
        """
        Compute the agent-specific loss.

        Each agent defines its own objective function, enabling heterogeneous
        learning strategies within the same system.

        Parameters
        ----------
        batch : tuple
            Batch containing:
            - embA : torch.Tensor
                Embeddings of Anchor samples.
            - embP : torch.Tensor
                Embeddings of Positive samples.
            - embN : torch.Tensor | None
                Embeddings of Negative samples
                (None when using contrastive loss).
            - y : torch.Tensor
                Target tensor used by the Siamese loss.


        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss.
        """
        embA, embP, embN, y = batch
        loss = self.siamese(z1=embA, z2=embP, z3=embN, y=y)

        return loss


if __name__ == '__main__':
    pass
