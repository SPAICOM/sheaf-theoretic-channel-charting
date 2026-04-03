"""Plot latent spaces for flat bundle (BS 0 and BS 1 - partially overlapping).

This script creates:
1. Plot for latent spaces of private (local) datasets of the two base stations
2. Plot of the latent representations for the shared_dataset between these two base stations
3. A final plot with the map applied to the latent representations of the shared_dataset
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

matplotlib.use('Agg')

AGENT_COLORS = plt.cm.tab10.colors
BS0_COLOR = AGENT_COLORS[0]
BS1_COLOR = AGENT_COLORS[1]

RESULTS_DIR = Path('results/4_agents/flat_bundle')
CHECKPOINT_PATH = Path(
    'checkpoints/4_agents/sheaf_cc_night_flat_bundle_2026-04-02_15-45-54_loss0.9593_epochs5_folds4.pt'
)
OUTPUT_DIR = Path('imgs/flat_bundle_bs01')


def _compute_path_distance(pos):
    """Compute cumulative path distance along trajectory."""
    pos_np = pos.detach().cpu().numpy()
    diffs = np.diff(pos_np, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    return cumulative


def _compute_spatial_gradient(pos):
    """Compute spatial gradient based on distance from origin."""
    pos_np = pos.detach().cpu().numpy()
    return np.sqrt(pos_np[:, 0] ** 2 + pos_np[:, 1] ** 2)


def _plot_convex_hull(ax, points, color, label, alpha=0.2, linewidth=2):
    """Plot a convex hull around points with dotted contour (no scatter points)."""
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return

    points_np = points.detach().cpu().numpy()

    try:
        hull = ConvexHull(points_np)
        hull_points = points_np[hull.vertices]

        ax.plot(
            np.append(hull_points[:, 0], hull_points[0, 0]),
            np.append(hull_points[:, 1], hull_points[0, 1]),
            color=color,
            linestyle='--',
            linewidth=linewidth,
            alpha=0.8,
            label=label,
        )
    except Exception:
        pass


def _scatter_colored(
    ax, embs, path_dist, cmap, norm, label, alpha=0.5, s=5, shading_color=None, shading_alpha=0.3
):
    embs_np = embs.detach().cpu().numpy()

    if shading_color is not None:
        ax.scatter(
            embs_np[:, 0],
            embs_np[:, 1],
            s=s * 1.5,
            alpha=shading_alpha,
            color=shading_color,
            rasterized=True,
        )

    scatter = ax.scatter(
        embs_np[:, 0],
        embs_np[:, 1],
        s=s,
        alpha=alpha,
        c=path_dist,
        cmap=cmap,
        norm=norm,
        label=label,
        rasterized=True,
    )

    return scatter


def _scatter_solid(ax, embs, color, label, alpha=0.5, s=5):
    embs_np = embs.detach().cpu().numpy()
    ax.scatter(
        embs_np[:, 0], embs_np[:, 1], s=s, alpha=alpha, color=color, label=label, rasterized=True
    )


def _find_shared_indices(local_embs, shared_embs):
    """Find indices of shared samples within local dataset by matching embeddings."""
    local_np = local_embs.detach().cpu().numpy()
    shared_np = shared_embs.detach().cpu().numpy()

    local_strs = [','.join(f'{x:.6f}' for x in row) for row in local_np]
    shared_strs = [','.join(f'{x:.6f}' for x in row) for row in shared_np]

    local_set = set(local_strs)
    indices = []
    for s in shared_strs:
        if s in local_set:
            idx = local_strs.index(s)
            indices.append(idx)

    return indices


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading checkpoint for local_reference_frames...')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    orchestrator = checkpoint
    orchestrator.eval()
    local_reference_frames = orchestrator.local_reference_frames

    print('Loading local agent data...')
    local_0 = torch.load(RESULTS_DIR / 'local_agent_0.pt', weights_only=False)
    local_1 = torch.load(RESULTS_DIR / 'local_agent_1.pt', weights_only=False)

    print('Loading shared dataset (BS 0 and BS 1)...')
    shared = torch.load(RESULTS_DIR / 'shared_0_1.pt', weights_only=False)

    embs_0 = local_0['embs']
    embs_1 = local_1['embs']
    pos_0 = local_0['pos']
    pos_1 = local_1['pos']
    shared_embs_0 = shared['embs_i']
    shared_embs_1 = shared['embs_j']

    print('Finding shared sample indices in local datasets...')
    shared_idx_0 = _find_shared_indices(embs_0, shared_embs_0)
    shared_idx_1 = _find_shared_indices(embs_1, shared_embs_1)
    print(f'  BS 0: {len(shared_idx_0)} / {embs_0.shape[0]} samples are shared')
    print(f'  BS 1: {len(shared_idx_1)} / {embs_1.shape[0]} samples are shared')

    private_idx_0 = [i for i in range(len(embs_0)) if i not in shared_idx_0]
    private_idx_1 = [i for i in range(len(embs_1)) if i not in shared_idx_1]

    print(f'BS 0 private: {len(private_idx_0)} samples')
    print(f'BS 1 private: {len(private_idx_1)} samples')
    print(f'Shared dataset: {shared_embs_0.shape[0]} samples per base station')

    print('Computing spatial gradients for coloring...')
    grad_0 = _compute_spatial_gradient(pos_0)
    grad_1 = _compute_spatial_gradient(pos_1)

    grad_max = max(grad_0.max(), grad_1.max())
    norm = Normalize(vmin=0, vmax=grad_max)
    cmap = 'viridis'

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.scatter(
        pos_0[:, 0].detach().cpu().numpy(),
        pos_0[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.5,
        c=grad_0,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, pos_0[shared_idx_0], BS0_COLOR, 'shared', linewidth=3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', 'box')
    ax.set_title('BS 0 - Ground Truth Positions')

    ax = axes[0, 1]
    ax.scatter(
        pos_1[:, 0].detach().cpu().numpy(),
        pos_1[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.5,
        c=grad_1,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, pos_1[shared_idx_1], BS1_COLOR, 'shared', linewidth=3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', 'box')
    ax.set_title('BS 1 - Ground Truth Positions')

    ax = axes[1, 0]
    ax.scatter(
        embs_0[:, 0].detach().cpu().numpy(),
        embs_0[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        c=grad_0,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, embs_0[shared_idx_0], BS0_COLOR, 'shared', linewidth=3)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_aspect('equal', 'box')
    ax.legend(markerscale=3)
    ax.set_title('BS 0 - Latent Space')

    ax = axes[1, 1]
    ax.scatter(
        embs_1[:, 0].detach().cpu().numpy(),
        embs_1[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        c=grad_1,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, embs_1[shared_idx_1], BS1_COLOR, 'shared', linewidth=3)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_aspect('equal', 'box')
    ax.legend(markerscale=3)
    ax.set_title('BS 1 - Latent Space')

    fig.suptitle('Flat Bundle - Ground Truth vs Latent Spaces (BS 0 & BS 1)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'latent_spaces_raw.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "latent_spaces_raw.png"}')

    R_0 = local_reference_frames['0'].cpu()
    R_1 = local_reference_frames['1'].cpu()
    shared_aligned_0 = shared_embs_0 @ R_0.T
    shared_aligned_1 = shared_embs_1 @ R_1.T

    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_solid(ax, shared_aligned_0, BS0_COLOR, f'BS 0 (aligned)', alpha=0.6, s=10)
    _scatter_solid(ax, shared_aligned_1, BS1_COLOR, f'BS 1 (aligned)', alpha=0.6, s=10)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_aspect('equal', 'box')
    ax.legend(markerscale=3)
    ax.set_title('Shared Dataset - Aligned Latent Space\n(Map applied: emb @ R.T)')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'shared_aligned.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "shared_aligned.png"}')

    print('\nDone!')


if __name__ == '__main__':
    main()
