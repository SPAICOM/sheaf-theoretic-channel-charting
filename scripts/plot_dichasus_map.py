"""Plot UE trajectories and base station coverage for DICHASUS cf0x datasets.

Coverage areas are derived from the same k-means spatial masking logic used in
the DICHASUSDataModule (four-single-group case [[0],[1],[2],[3]]):

  - BS 1 (phys 0) and BS 3 (phys 2): lower branch — k-means ranks 1+2
  - BS 2 (phys 1) and BS 4 (phys 3): upper branch — k-means ranks 0+1

Because BSs within the same branch see identical point sets, a small
low-frequency random noise field is added to the smoothed density before
contouring so adjacent shadows never perfectly coincide.

Output: results/dichasus_map.png
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import gaussian_filter

matplotlib.use('Agg')

STYLE_PATH = Path(__file__).resolve().parents[1] / 'config/plotting/plt.mplstyle'
plt.style.use(str(STYLE_PATH))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physical BS positions (x, y) in metres — DICHASUS cf0x antenna array centres.
# Array i+1  ↔  physical BS id i  ↔  displayed label "BS i+1".
BS_POSITIONS = np.array([
    [ 2.6747, -13.8973],   # phys 0 → BS 1
    [-11.2503,  -9.6890],  # phys 1 → BS 2
    [ -1.5314, -15.0595],  # phys 2 → BS 3
    [-12.6844,  -4.4833],  # phys 3 → BS 4
])

BS_COLORS  = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']  # red, green, blue, orange
BS_LABELS  = [f'BS {i + 1}' for i in range(4)]             # BS 1 … BS 4

# Annotation nudges (dx, dy) so labels don't overlap the marker
BS_ANNOTATION_OFFSET = [
    ( 0.4,  0.6),   # BS 1
    (-2.5,  0.6),   # BS 2
    ( 0.4, -1.0),   # BS 3
    ( 0.4,  0.6),   # BS 4
]

# Coverage rank sets — mirrors DICHASUSDataModule four-single-group logic:
#   lower_phys = {0, 2}  (BS 1, BS 3 — bottom of the hall)
#       → ranks 1 + 2  (middle + lower-right arm)
#   upper_phys = {1, 3}  (BS 2, BS 4 — upper-left of the hall)
#       → ranks 0 + 1  (upper-left arm + middle)
BS_COVERAGE_RANKS = {
    0: [1, 2],   # BS 1 (phys 0, lower-right) — lower branch
    1: [0, 1],   # BS 2 (phys 1, upper-left)  — upper branch
    2: [1, 2],   # BS 3 (phys 2, lower)        — lower branch
    3: [0, 1],   # BS 4 (phys 3, upper-left)   — upper branch
}

# Building polygon — rectangular obstacle in the inner corner of the L-shape,
# slightly rotated and enlarged horizontally to hug the trajectory boundary.
# The region x < -5, y < -10 has zero trajectory points → clean empty space.
_bc = np.array([-7.5, -11.5])          # building centre
_bw, _bh = 3.9, 1.85                   # half-width / half-height in metres
_theta = np.radians(15)                 # CCW rotation
_R = np.array([[np.cos(_theta), -np.sin(_theta)],
               [np.sin(_theta),  np.cos(_theta)]])
BUILDING_CENTER   = _bc
BUILDING_VERTICES = np.array([
    [-_bw, -_bh], [_bw, -_bh], [_bw, _bh], [-_bw, _bh],
]) @ _R.T + _bc

DATA_DIR   = Path('dichasus')
FILE_STEMS = ['dichasus-cf02', 'dichasus-cf03', 'dichasus-cf04']
OUT_PATH   = Path('results/dichasus_map.png')

# Density grid resolution and smoothing
GRID_RES        = 300
SMOOTH_SIGMA    = 11    # Gaussian sigma in grid cells for main density
NOISE_SIGMA     = 30    # sigma for the low-frequency perturbation field
NOISE_AMPLITUDE = 0.07  # fraction of peak density added as per-BS noise
CONTOUR_THRESH  = 0.10  # density fraction below which coverage is not shown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectories(stems: list[str]) -> np.ndarray:
    parts = []
    for stem in stems:
        p = DATA_DIR / f'{stem}_pos.npy'
        if p.exists():
            parts.append(np.load(str(p), mmap_mode='r').copy())
        else:
            print(f'[warn] {p} not found — skipping.')
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 2))


def compute_point_ranks(pos: np.ndarray) -> np.ndarray:
    """K-means (k=3) exactly as in DICHASUSDataModule._compute_spatial_clusters_3."""
    centroids, labels = kmeans2(
        pos.astype(np.float32), 3, seed=42, iter=50, minit='points'
    )
    diag_scores = centroids[:, 0] - 0.852 * centroids[:, 1]
    rank_order = np.argsort(diag_scores)
    rank_of_cluster = np.empty(3, dtype=int)
    for rank, cluster_id in enumerate(rank_order):
        rank_of_cluster[cluster_id] = rank
    return rank_of_cluster[labels]


def _bridge_density(
    bs_pos: np.ndarray,
    cov_pts: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    sigma_tube_m: float = 0.7,
    amplitude: float = CONTOUR_THRESH * 1.6,
) -> np.ndarray:
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
    tube = amplitude * np.exp(-dist**2 / (2.0 * (sigma_tube_m / cell_m)**2 * cell_m**2))
    return tube.reshape(xg.shape)


def coverage_density(
    pts: np.ndarray,
    bs_pos: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smoothed 2-D density with bridge to BS and a per-BS random perturbation."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pos_all = load_trajectories(FILE_STEMS)
    print(f'Loaded {len(pos_all):,} UE positions from {len(FILE_STEMS)} files.')

    point_ranks = compute_point_ranks(pos_all)
    print('K-means spatial clustering done.')

    margin = 1.5
    x_min = min(pos_all[:, 0].min(), BS_POSITIONS[:, 0].min()) - margin
    x_max = max(pos_all[:, 0].max(), BS_POSITIONS[:, 0].max()) + margin
    y_min = min(pos_all[:, 1].min(), BS_POSITIONS[:, 1].min()) - margin
    y_max = max(pos_all[:, 1].max(), BS_POSITIONS[:, 1].max()) + margin

    xg, yg = np.meshgrid(
        np.linspace(x_min, x_max, GRID_RES),
        np.linspace(y_min, y_max, GRID_RES),
    )

    fig, ax = plt.subplots(figsize=(11, 11))

    # --- 1. Coverage regions -------------------------------------------------
    for phys_id, (color, label) in enumerate(zip(BS_COLORS, BS_LABELS)):
        allowed_ranks = BS_COVERAGE_RANKS[phys_id]
        mask = np.isin(point_ranks, allowed_ranks)
        bs_pts = pos_all[mask]

        rng = np.random.default_rng(seed=phys_id * 43 + 11)
        density = coverage_density(bs_pts, BS_POSITIONS[phys_id], xg, yg, rng)

        ax.contourf(
            xg, yg, density,
            levels=[CONTOUR_THRESH, density.max()],
            colors=[color], alpha=0.18, zorder=1,
        )
        ax.contour(
            xg, yg, density,
            levels=[CONTOUR_THRESH],
            colors=[color], alpha=0.55, linewidths=1.1, zorder=2,
        )

    # --- 2. Building polygon -------------------------------------------------
    building = mpatches.Polygon(
        BUILDING_VERTICES, closed=True,
        facecolor='#bbbbbb', edgecolor='#444444',
        linestyle='--', linewidth=1.5, alpha=0.35, zorder=4,
    )
    ax.add_patch(building)
    ax.text(
        BUILDING_CENTER[0], BUILDING_CENTER[1], 'scatterer',
        ha='center', va='center', fontsize=20,
        color='#333333', zorder=5,
    )

    # --- 3. UE trajectories --------------------------------------------------
    ax.scatter(
        pos_all[:, 0], pos_all[:, 1],
        s=1.2, color='#333333', alpha=0.28,
        rasterized=True, zorder=3, label='UE trajectory',
    )

    # --- 4. Base station markers ---------------------------------------------
    for phys_id, (bs_pos, color, label, (dx, dy)) in enumerate(
        zip(BS_POSITIONS, BS_COLORS, BS_LABELS, BS_ANNOTATION_OFFSET)
    ):
        ax.scatter(
            bs_pos[0], bs_pos[1],
            s=250, color=color, marker='^',
            edgecolors='black', linewidths=0.8, zorder=5,
        )
        ax.annotate(
            label, xy=bs_pos,
            xytext=(bs_pos[0] + dx, bs_pos[1] + dy),
            fontweight='bold', color='black', zorder=6,
        )

    # --- 5. Legend -----------------------------------------------------------
    traj_handle = mlines.Line2D(
        [], [], linestyle='-', linewidth=1.5,
        color='#333333', alpha=0.6,
        label='UE trajectory',
    )
    bs_handles = [
        mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.55, label=f'{lbl} coverage')
        for c, lbl in zip(BS_COLORS, BS_LABELS)
    ]
    building_handle = mpatches.Patch(
        facecolor='#bbbbbb', edgecolor='#444444',
        linestyle='--', linewidth=1.5, label='Building',
    )
    ax.legend(
        handles=[traj_handle] + bs_handles + [building_handle],
        loc='lower left',
    )

    # --- 6. Formatting -------------------------------------------------------
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal', 'box')
    ax.grid(False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {OUT_PATH}')


if __name__ == '__main__':
    main()
