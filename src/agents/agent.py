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
import torch.nn.functional as F

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
    dropout : float, default=0.0
        Dropout probability applied after each hidden linear layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
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
        self.dropout_layers = nn.ModuleList()

        dim = in_dim

        # Build hidden layers that progressively reduce dimensionality
        for _ in range(self.num_hidden_layers - 1):
            self.linear_layers.append(nn.Linear(dim, dim // 2))
            self.norm_layers.append(nn.BatchNorm1d(dim // 2))
            self.dropout_layers.append(nn.Dropout(p=dropout))
            dim = dim // 2

        # Final projection layer producing the embedding
        self.linear_layers.append(nn.Linear(dim, self.out_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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
        # Track whether input is 3D: (BS, P, d) — e.g. multiple positives/neg
        shape_3d = x.dim() == 3
        if shape_3d:
            BS, P, _ = x.shape
            x = x.reshape(BS * P, -1)

        # Apply BatchNorm + activation + Dropout to all layers except the last
        for i in range(self.num_hidden_layers - 1):
            x = self.norm_layers[i](self.linear_layers[i](x))
            x = self.act(x)
            x = self.dropout_layers[i](x)

        # Final linear projection (no activation)
        x = self.linear_layers[-1](x)

        if shape_3d:
            x = x.reshape(BS, P, -1)

        return x


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
    dropout : float, default=0.0
        Dropout probability applied after each hidden linear layer.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 256,
        num_hidden_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        # Activation function
        self.act = nn.ReLU()

        # Container for decoder layers
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        dim = in_dim

        # Hidden layers progressively expand dimensionality
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(dim, dim * 2))
            self.norm_layers.append(nn.BatchNorm1d(dim * 2))
            self.dropout_layers.append(nn.Dropout(p=dropout))
            dim = dim * 2

        # Final reconstruction layer
        self.layers.append(nn.Linear(dim, self.out_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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

        shape_3d = x.dim() == 3
        if shape_3d:
            BS, P, _ = x.shape
            x = x.reshape(BS * P, -1)

        # Apply BatchNorm + activation + Dropout to all layers except the last
        for i in range(self.num_hidden_layers - 1):
            x = self.norm_layers[i](self.layers[i](x))
            x = self.act(x)
            x = self.dropout_layers[i](x)

        if shape_3d:
            x = x.reshape(BS, P, -1)

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
    use_decoder : bool, default=True
        Whether to include the reconstruction loss (normalised MSE on the
        anchor) on top of the triplet loss.
    margin : float, default=1.0
        Margin parameter used in Siamese losses.
    lr : float, default=1e-3
        Learning rate for the optimizer.
    dropout : float, default=0.0
        Dropout probability applied after each hidden linear layer in the
        encoder and decoder.
    """

    def __init__(
        self,
        in_dim: int,
        distance_mode: str,
        loss_mode: str,
        out_dim: int = 2,
        num_hidden_layers: int = 6,
        use_decoder: bool = True,
        margin: float = 1.0,
        lr: float = 1e-3,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate configuration
        assert distance_mode in {'euclidean', 'euclidean2', 'cosine'}, (
            'Provide a valid distance mode'
        )
        assert loss_mode in {'contrastive', 'triplet'}, 'Provide a valid layer mode'

        self.use_decoder = use_decoder
        self.distance_mode = distance_mode
        self.loss_mode = loss_mode

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.margin = margin
        self.lr = lr

        self.encoder = Encoder(
            in_dim=in_dim,
            out_dim=out_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )

        self.decoder = (
            Decoder(
                in_dim=out_dim,
                out_dim=in_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
            )
            if use_decoder
            else None
        )

        self.siamese = SiameseLayer(
            loss_mode=loss_mode,
            distance_mode=distance_mode,
            margin=margin,
        )

    def forward(
        self,
        batch,
        triplet_mode: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
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
            - pos : torch.Tensor
                Anchor samples positions.

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
            xA, xP, xN, y, _ = batch

            # Compute embeddings
            embA = self.encoder(xA)
            embP = self.encoder(xP)

            # Negative sample may be None in contrastive mode
            embN = self.encoder(xN) if xN is not None else None

            out = (embA, embP, embN, y)

        else:
            out = self.encoder(batch)
        return out

    def compute_loss(
        self,
        batch,
        embeddings,
    ) -> torch.Tensor:
        """
        Compute the agent-specific loss: triplet loss + reconstruction loss.

        Each agent defines its own objective function, enabling heterogeneous
        learning strategies within the same system.

        Parameters
        ----------
        batch : tuple
            Raw input batch containing:
            - xA : torch.Tensor
                Anchor samples (used as reconstruction target when
                use_decoder=True).
            - xP : torch.Tensor
                Positive samples (unused here).
            - xN : torch.Tensor | None
                Negative samples (unused here).
            - y : torch.Tensor
                Target tensor (unused here, taken from embeddings).
        embeddings : tuple
            Pre-computed encoder outputs containing:
            - embA : torch.Tensor
                Embeddings of anchor samples.
            - embP : torch.Tensor
                Embeddings of positive samples.
            - embN : torch.Tensor | None
                Embeddings of negative samples.
            - y : torch.Tensor
                Target tensor used by the Siamese loss.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys 'triplet_loss' and, when use_decoder=True,
            'rec_loss'.
        """
        xA, _, _, _, _ = batch
        embA, embP, embN, y = embeddings

        losses = {'triplet_loss': self.siamese(z1=embA, z2=embP, z3=embN, y=y)}

        if self.use_decoder:
            losses['rec_loss'] = F.mse_loss(self.decoder(embA), xA) / self.in_dim

        return losses


if __name__ == '__main__':
    pass
