import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceLayer(nn.Module):
    def __init__(self, distance_mode: str) -> None:
        super().__init__()
        assert distance_mode in {'euclidean', 'cosine'}
        self.distance_mode = distance_mode

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        match self.distance_mode:
            case 'euclidean':
                z1 = z1.unsqueeze(1)  # (B, 1, D)
                dist = (z1 - z2).pow(2).sum(-1)  # (B, K)
                print(f'{dist.shape=}')
                return dist.sum(dim=1)  # (B,)

            case 'cosine':
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                z1 = z1.unsqueeze(1)  # (B, 1, D)
                sim = (z1 * z2).sum(-1)  # (B, K)
                dist = 1 - sim  # (B, K)
                return dist.sum(dim=1)  # (B,)


# ---- TEST SCRIPT ----
if __name__ == '__main__':
    torch.manual_seed(0)

    B = 4  # batch size
    K = 3  # number of comparisons per sample
    D = 5  # embedding dimension

    # Anchor embeddings
    z1 = torch.randn(B, D)

    # Positive/negative embeddings
    z2 = torch.randn(B, K, D)

    # Euclidean distance
    euclidean_layer = DistanceLayer('euclidean')
    euclidean_out = euclidean_layer(z1, z2)

    # Cosine distance
    cosine_layer = DistanceLayer('cosine')
    cosine_out = cosine_layer(z1, z2)

    print('z1 shape:', z1.shape)
    print('z2 shape:', z2.shape)
    print()

    print('Euclidean output shape:', euclidean_out.shape)
    print('Euclidean output:', euclidean_out)
    print()

    print('Cosine output shape:', cosine_out.shape)
    print('Cosine output:', cosine_out)
