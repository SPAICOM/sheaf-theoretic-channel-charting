"""Plot ground truth trajectory with coverage areas for BS 0 only."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# Load plotting style
STYLE_PATH = Path(
    '/home/engrima/projects/SheafTheoreticChannelCharting/config/plotting/plt.mplstyle'
)
plt.style.use(str(STYLE_PATH))

matplotlib.use('Agg')

AGENT_COLORS = plt.cm.tab10.colors
BS0_COLOR = AGENT_COLORS[0]

RESULTS_DIR = Path('results/4_agents/flat_bundle')
OUTPUT_DIR = Path('imgs/flat_bundle_bs01')

BS_A = 0

# BS positions from plot_dichasus_map.py
BS_POSITIONS = {
    0: np.array([2.6747, -13.8973]),
    1: np.array([-11.2503, -9.6890]),
    2: np.array([-1.5314, -15.0595]),
    3: np.array([-12.6844, -4.4833]),
}

# Density grid parameters
GRID_RES = 200
SMOOTH_SIGMA = 11
NOISE_SIGMA = 30
NOISE_AMPLITUDE = 0.07
CONTOUR_THRESH = 0.10


def _bridge_density(bs_pos, cov_pts, xg, yg, sigma_tube_m=0.7, amplitude=0.16):
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


def _plot_convex_hull(ax, points, color, linewidth=2):
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
        )
    except Exception:
        pass


def _find_shared_indices(local_embs, shared_embs):
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

    print('Loading local agent data...')
    local_a = torch.load(RESULTS_DIR / f'local_agent_{BS_A}.pt', weights_only=False)

    print(f'Loading shared dataset (BS {BS_A})...')
    shared = torch.load(RESULTS_DIR / 'shared_0_1.pt', weights_only=False)

    embs_a = local_a['embs']
    pos_a = local_a['pos']
    shared_embs_a = shared['embs_i']

    print('Finding shared sample indices...')
    shared_idx_a = _find_shared_indices(embs_a, shared_embs_a)
    print(f'  BS {BS_A}: {len(shared_idx_a)} / {embs_a.shape[0]} samples are shared')

    bs_pos_a = BS_POSITIONS[BS_A]

    pos_a_np = pos_a.detach().cpu().numpy()

    margin = 2.0
    x_min, x_max = pos_a_np[:, 0].min() - margin, pos_a_np[:, 0].max() + margin
    y_min, y_max = pos_a_np[:, 1].min() - margin, pos_a_np[:, 1].max() + margin

    xg, yg = np.meshgrid(
        np.linspace(x_min, x_max, GRID_RES),
        np.linspace(y_min, y_max, GRID_RES),
    )

    rng_a = np.random.default_rng(seed=BS_A * 43 + 11)
    density_a = _coverage_density(pos_a_np, bs_pos_a, xg, yg, rng_a)

    CONTOUR_COLOR = 'indigo'

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.contour(
        xg, yg, density_a, levels=[CONTOUR_THRESH], colors=[BS0_COLOR], linewidths=2.5, alpha=0.7
    )

    grad_a = np.sqrt(pos_a_np[:, 0] ** 2 + pos_a_np[:, 1] ** 2)
    norm = Normalize(vmin=0, vmax=grad_a.max())
    cmap = 'plasma'

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

    _plot_convex_hull(ax, pos_a[shared_idx_a], CONTOUR_COLOR, linewidth=6)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

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
            edgecolor=CONTOUR_COLOR,
            linestyle='--',
            linewidth=3,
            label='Shared region',
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        ncol=2,
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
    )

    fig.savefig(OUTPUT_DIR / 'ground_truth_bs0.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / "ground_truth_bs0.pdf"}')

    print('\nDone!')


if __name__ == '__main__':
    main()
