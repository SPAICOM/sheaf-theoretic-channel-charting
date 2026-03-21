"""
Wrappers for distance and loss computation for Siamese networks.
Supports contrastive and triplet losses with Euclidean or cosine distances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceLayer(nn.Module):
    """
    Compute pairwise distances between embeddings.

    Supports 'euclidean' and 'cosine' distance modes.
    """

    def __init__(
        self,
        distance_mode: str,
    ) -> None:
        """
        Initialize the distance layer.

        Parameters
        ----------
        distance_mode : str
            Distance metric: 'euclidean' or 'cosine'.
        """
        super().__init__()
        assert distance_mode in {'euclidean', 'cosine'}, (
            'Provide a valid distance mode'
        )

        self.distance_mode = distance_mode

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Compute the distance between two batches of embeddings.

        Parameters
        ----------
        z1, z2 : torch.Tensor
            Tensors of shape (batch_size, embedding_dim).

        Returns
        -------
        torch.Tensor | None
            Pairwise distance for each batch element.
        """
        match self.distance_mode:
            case 'euclidean':
                # Anchor expansion
                z1 = z1.unsqueeze(1)  # (B, 1, D)

                # Distance computation (batching + broadcasting)
                dist = (z1 - z2).pow(2).sum(dim=-1)  # (B, K)

                # Loss aggregation
                loss = dist.sum(dim=1).mean()
                return loss

            case 'cosine':
                # Normalize
                z1 = F.normalize(z1, dim=-1)  # (B, D)
                z2 = F.normalize(z2, dim=-1)  # (B, K, D)

                # Anchor expansion
                z1 = z1.unsqueeze(1)  # (B, 1, D)

                # Cosine similarity for each positive
                sim = (z1 * z2).sum(dim=-1)  # (B, K)

                # Convert to cosine distance
                dist = 1 - sim  # (B, K)

                # Loss aggregation
                loss = dist.sum(dim=1).mean()
                return loss

            case _:
                return None


class LossLayer(nn.Module):
    """
    Compute contrastive or triplet loss using a distance layer.
    """

    def __init__(
        self,
        loss_mode: str,
        distance_mode: str,
        margin: float = 1.0,
    ) -> None:
        """
        Initialize the loss layer.

        Parameters
        ----------
        loss_mode : str
            'contrastive' or 'triplet'.
        distance_mode : str
            'euclidean' or 'cosine'.
        margin : float
            Margin for contrastive/triplet loss.
        """
        super().__init__()
        assert loss_mode in {'contrastive', 'triplet'}, (
            'Provide a valid layer mode'
        )
        assert distance_mode in {'euclidean', 'cosine'}, (
            'Provide a valid distance mode'
        )

        self.margin = margin
        self.loss_mode = loss_mode
        self.dist_layer = DistanceLayer(distance_mode=distance_mode)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
        z3: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the loss for a batch.

        Parameters
        ----------
        z1, z2 : torch.Tensor
            Anchor and positive embeddings.
        z3 : torch.Tensor | None
            Negative embeddings (for triplet loss).
        y : torch.Tensor
            Labels (0/1 for contrastive loss).

        Returns
        -------
        torch.Tensor
            Mean loss over the batch.
        """
        match self.loss_mode:
            case 'contrastive':
                # Contrastive loss
                assert y is not None
                y = y.float()

                d = self.dist_layer(z1, z2)
                dP = y * d.pow(2)
                dN = (1 - y) * torch.clamp(self.margin - d, min=0.0).pow(2)

                L = dP + dN

            case 'triplet':
                # Triplet loss

                assert z3 is not None

                dAP = self.dist_layer(z1, z2)
                dAN = self.dist_layer(z1, z3)
                # print(dAP, dAN)
                L = torch.clamp(dAP - dAN + self.margin, min=0.0)

            case _:
                raise RuntimeError(
                    'The passed loss_mode is not currently supported.'
                    'Chose between "contrastive" or "triplet".'
                )

        return L.mean()


class SiameseLayer(nn.Module):
    """
    High-level wrapper combining distance and loss computation
    for Siamese networks.
    """

    def __init__(
        self,
        loss_mode: str,
        distance_mode: str,
        margin: float,
    ) -> None:
        """
        Initialize the Siamese loss layer.

        Parameters
        ----------
        loss_mode : str
            'contrastive' or 'triplet'.
        distance_mode : str
            'euclidean' or 'cosine'.
        margin : float
            Margin for loss computation.
        """
        super().__init__()
        assert loss_mode in ['contrastive', 'triplet'], (
            'Provide a valid layer mode'
        )
        assert distance_mode in ['euclidean', 'cosine'], (
            'Provide a valid distance mode'
        )

        self.loss_mode = loss_mode
        self.distance_mode = distance_mode
        self.margin = margin

        # Build the underlying LossLayer
        self.loss_func = LossLayer(
            margin=margin,
            loss_mode=loss_mode,
            distance_mode=distance_mode,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
        z3: torch.Tensor | None = None,
    ) -> None:
        """
        Compute the Siamese loss for the given embeddings.

        Parameters
        ----------
        z1, z2 : torch.Tensor
            Anchor and positive embeddings.
        z3 : torch.Tensor | None
            Negative embeddings (only for triplet loss).
        y : torch.Tensor
            Labels (0/1 for contrastive loss).

        Returns
        -------
        torch.Tensor
            Mean loss over the batch.
        """
        return self.loss_func(z1=z1, z2=z2, z3=z3, y=y)


if __name__ == '__main__':
    pass
