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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.cluster.vq import kmeans2
from scipy.ndimage import gaussian_filter

# Load plotting style
STYLE_PATH = Path(
    '/home/engrima/projects/SheafTheoreticChannelCharting/config/plotting/plt.mplstyle'
)
plt.style.use(str(STYLE_PATH))

matplotlib.use('Agg')

AGENT_COLORS = plt.cm.tab10.colors
BS0_COLOR = AGENT_COLORS[0]
BS1_COLOR = AGENT_COLORS[1]

RESULTS_DIR = Path('results/4_agents/flat_bundle')
CHECKPOINT_PATH = Path(
    'checkpoints/4_agents/sheaf_cc_night_flat_bundle_2026-04-02_15-45-54_loss0.9593_epochs5_folds4.pt'
)
OUTPUT_DIR = Path('imgs/flat_bundle_bs01')

# User asked for agents 0 and 1
BS_A = 0
BS_B = 1

# BS positions from plot_dichasus_map.py
BS_POSITIONS = {
    0: np.array([2.6747, -13.8973]),
    1: np.array([-11.2503, -9.6890]),
    2: np.array([-1.5314, -15.0595]),
    3: np.array([-12.6844, -4.4833]),
}

# Coverage rank sets - mirrors DICHASUSDataModule four-single-group logic
BS_COVERAGE_RANKS = {
    0: [1, 2],
    1: [0, 1],
    2: [1, 2],
    3: [0, 1],
}

# Density grid parameters
GRID_RES = 200
SMOOTH_SIGMA = 11
NOISE_SIGMA = 30
NOISE_AMPLITUDE = 0.07
CONTOUR_THRESH = 0.10


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


def _bridge_density(bs_pos, cov_pts, xg, yg, sigma_tube_m=0.7, amplitude=0.16):
    """Gaussian tube from the BS position to its nearest coverage point."""
    nearest_pt = cov_pts[np.argmin(np.linalg.norm(cov_pts - bs_pos, axis=1))]
    seg = nearest_pt - bs_pos
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        return np.zeros_like(xg)

    seg_dir = seg / seg_len
    x_range = xg[0, -1] - xg[0, 0]
    cell_m = x_range / xg.shape[1]

    gx, gy = xg.ravel(), yg.ravel()
    v = np.column_stack([gx - bs_pos[0], gy - bs_pos[1]])
    t = np.clip(v @ seg_dir, 0.0, seg_len)
    closest = bs_pos + np.outer(t, seg_dir)
    dist = np.linalg.norm(np.column_stack([gx, gy]) - closest, axis=1)
    tube = amplitude * np.exp(-(dist**2) / (2.0 * (sigma_tube_m / cell_m) ** 2 * cell_m**2))
    return tube.reshape(xg.shape)


def _coverage_density(pts, bs_pos, xg, yg, rng):
    """Compute smoothed 2-D density with bridge to BS and a per-BS random perturbation."""
    ny, nx = xg.shape
    x_edges = np.linspace(xg.min(), xg.max(), nx + 1)
    y_edges = np.linspace(yg.min(), yg.max(), ny + 1)

    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    H = H.T.astype(float)
    H = gaussian_filter(H, sigma=SMOOTH_SIGMA)
    if H.max() > 0:
        H /= H.max()

    raw_noise = rng.standard_normal((ny, nx))
    smooth_noise = gaussian_filter(raw_noise, sigma=NOISE_SIGMA)
    smooth_noise /= np.abs(smooth_noise).max() + 1e-9
    H = H + smooth_noise * NOISE_AMPLITUDE
    H = H + _bridge_density(bs_pos, pts, xg, yg)
    return H


def _compute_point_ranks(pos):
    """K-means (k=3) exactly as in DICHASUSDataModule._compute_spatial_clusters_3."""
    centroids, labels = kmeans2(pos.astype(np.float32), 3, seed=42, iter=50, minit='points')
    diag_scores = centroids[:, 0] - 0.852 * centroids[:, 1]
    rank_order = np.argsort(diag_scores)
    rank_of_cluster = np.empty(3, dtype=int)
    for rank, cluster_id in enumerate(rank_order):
        rank_of_cluster[cluster_id] = rank
    return rank_of_cluster[labels]


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


def _plot_filled_hull(ax, points, color, alpha=0.2):
    """Plot a filled convex hull for coverage area."""
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return

    points_np = points.detach().cpu().numpy()

    try:
        hull = ConvexHull(points_np)
        hull_points = points_np[hull.vertices]
        polygon = np.vstack([hull_points, hull_points[0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)
    except Exception:
        pass


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
    local_a = torch.load(RESULTS_DIR / f'local_agent_{BS_A}.pt', weights_only=False)
    local_b = torch.load(RESULTS_DIR / f'local_agent_{BS_B}.pt', weights_only=False)

    print(f'Loading shared dataset (BS {BS_A} and BS {BS_B})...')
    shared = torch.load(RESULTS_DIR / f'shared_{BS_A}_{BS_B}.pt', weights_only=False)

    embs_a = local_a['embs']
    embs_b = local_b['embs']
    pos_a = local_a['pos']
    pos_b = local_b['pos']
    shared_embs_a = shared['embs_i']
    shared_embs_b = shared['embs_j']

    print('Finding shared sample indices in local datasets...')
    shared_idx_a = _find_shared_indices(embs_a, shared_embs_a)
    shared_idx_b = _find_shared_indices(embs_b, shared_embs_b)
    print(f'  BS {BS_A}: {len(shared_idx_a)} / {embs_a.shape[0]} samples are shared')
    print(f'  BS {BS_B}: {len(shared_idx_b)} / {embs_b.shape[0]} samples are shared')

    print(f'BS {BS_A} total: {embs_a.shape[0]} samples')
    print(f'BS {BS_B} total: {embs_b.shape[0]} samples')
    print(f'Shared dataset: {shared_embs_a.shape[0]} samples per base station')

    print('Computing spatial gradients for coloring...')
    grad_a = _compute_spatial_gradient(pos_a)
    grad_b = _compute_spatial_gradient(pos_b)

    grad_max = max(grad_a.max(), grad_b.max())
    norm = Normalize(vmin=0, vmax=grad_max)
    cmap = 'plasma'

    # BS positions
    bs_pos_a = BS_POSITIONS[BS_A]
    bs_pos_b = BS_POSITIONS[BS_B]

    # Compute density-based coverage contours
    pos_a_np = pos_a.detach().cpu().numpy()
    pos_b_np = pos_b.detach().cpu().numpy()

    # Determine grid bounds
    margin = 2.0
    all_pos = np.vstack([pos_a_np, pos_b_np])
    x_min, x_max = all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin
    y_min, y_max = all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin

    xg, yg = np.meshgrid(
        np.linspace(x_min, x_max, GRID_RES),
        np.linspace(y_min, y_max, GRID_RES),
    )

    # Compute densities for each BS - use ALL local points for full coverage
    rng_a = np.random.default_rng(seed=BS_A * 43 + 11)
    rng_b = np.random.default_rng(seed=BS_B * 43 + 11)

    density_a = _coverage_density(pos_a_np, bs_pos_a, xg, yg, rng_a)
    density_b = _coverage_density(pos_b_np, bs_pos_b, xg, yg, rng_b)

    CONTOUR_COLOR = 'indigo'

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Add row labels
    fig.text(0.02, 0.75, 'Ground Truth', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.25, 'Latent Space', fontsize=14, fontweight='bold', rotation=90, va='center')

    # BS positions
    bs_pos_a = BS_POSITIONS[BS_A]
    bs_pos_b = BS_POSITIONS[BS_B]

    ax = axes[0, 0]
    # Coverage area contour (density-based)
    ax.contour(
        xg, yg, density_a, levels=[CONTOUR_THRESH], colors=[BS0_COLOR], linewidths=2.5, alpha=0.7
    )
    ax.scatter(
        pos_a[:, 0].detach().cpu().numpy(),
        pos_a[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.5,
        c=grad_a,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.scatter(
        [bs_pos_a[0]],
        [bs_pos_a[1]],
        s=200,
        marker='*',
        color=BS0_COLOR,
        edgecolors='black',
        linewidths=1,
        zorder=10,
    )
    _plot_convex_hull(ax, pos_a[shared_idx_a], CONTOUR_COLOR, None, linewidth=3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    ax = axes[0, 1]
    ax.contour(
        xg, yg, density_b, levels=[CONTOUR_THRESH], colors=[BS1_COLOR], linewidths=2.5, alpha=0.7
    )
    ax.scatter(
        pos_b[:, 0].detach().cpu().numpy(),
        pos_b[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.5,
        c=grad_b,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.scatter(
        [bs_pos_b[0]],
        [bs_pos_b[1]],
        s=200,
        marker='*',
        color=BS1_COLOR,
        edgecolors='black',
        linewidths=1,
        zorder=10,
    )
    _plot_convex_hull(ax, pos_b[shared_idx_b], CONTOUR_COLOR, None, linewidth=3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    ax = axes[1, 0]
    ax.scatter(
        embs_a[:, 0].detach().cpu().numpy(),
        embs_a[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        c=grad_a,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, embs_a[shared_idx_a], CONTOUR_COLOR, None, linewidth=3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    ax = axes[1, 1]
    ax.scatter(
        embs_b[:, 0].detach().cpu().numpy(),
        embs_b[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        c=grad_b,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    _plot_convex_hull(ax, embs_b[shared_idx_b], CONTOUR_COLOR, None, linewidth=3)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor='none',
            edgecolor=BS0_COLOR,
            linewidth=2.5,
            alpha=0.7,
            label=f'BS {BS_A} coverage',
        ),
        mpatches.Patch(
            facecolor='none',
            edgecolor=BS1_COLOR,
            linewidth=2.5,
            alpha=0.7,
            label=f'BS {BS_B} coverage',
        ),
        mpatches.Patch(
            facecolor='none',
            edgecolor=CONTOUR_COLOR,
            linestyle='--',
            linewidth=3,
            label='Shared region',
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
    )

    fig.savefig(OUTPUT_DIR / 'latent_spaces_raw.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'latent_spaces_raw.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "latent_spaces_raw.pdf"}')
    print(f'Saved: {OUTPUT_DIR / "latent_spaces_raw.png"}')

    # Plot shared datasets before alignment (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    ax.scatter(
        shared_embs_a[:, 0].detach().cpu().numpy(),
        shared_embs_a[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        color=BS0_COLOR,
        rasterized=True,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    ax = axes[1]
    ax.scatter(
        shared_embs_b[:, 0].detach().cpu().numpy(),
        shared_embs_b[:, 1].detach().cpu().numpy(),
        s=3,
        alpha=0.6,
        color=BS1_COLOR,
        rasterized=True,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    fig.savefig(OUTPUT_DIR / 'shared_raw.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'shared_raw.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "shared_raw.pdf"}')
    print(f'Saved: {OUTPUT_DIR / "shared_raw.png"}')

    R_a = local_reference_frames[str(BS_A)].cpu()
    R_b = local_reference_frames[str(BS_B)].cpu()
    shared_aligned_a = shared_embs_a @ R_a.T
    shared_aligned_b = shared_embs_b @ R_b.T

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        shared_aligned_a[:, 0].detach().cpu().numpy(),
        shared_aligned_a[:, 1].detach().cpu().numpy(),
        s=10,
        alpha=0.6,
        color=BS0_COLOR,
        rasterized=True,
    )
    ax.scatter(
        shared_aligned_b[:, 0].detach().cpu().numpy(),
        shared_aligned_b[:, 1].detach().cpu().numpy(),
        s=10,
        alpha=0.6,
        color=BS1_COLOR,
        rasterized=True,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    fig.savefig(OUTPUT_DIR / 'shared_aligned.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'shared_aligned.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "shared_aligned.pdf"}')
    print(f'Saved: {OUTPUT_DIR / "shared_aligned.png"}')

    print('\nDone!')


if __name__ == '__main__':
    main()
