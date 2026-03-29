"""
Verification script for CSI / position index consistency.

Checks that the three layers of indexing in DeepMimoDataModule are mutually
consistent with the original DeepMIMO arrays, and that the triplet
sampling produces geometrically/spectrally sensible pairs:

  Check 1 — Position mapping
      dm.rx_pos_all_masked[gidx]
      == dm.rx_pos_all[dm.valid_idxs_all[gidx]][:2]

  Check 2 — Channel mapping (per BS)
      dm.bs_coords[bs_id]['channels'][gidx]
      == dm.ds[bs_id].channels[dm.valid_idxs_all[gidx]]

  Check 3 — Dataset __getitem__ round-trip (return_raw=True)
      H_A returned by dataset[i]
      == dm.ds[bs_id].channels[dm.valid_idxs_all[gidx]]
      pos returned by dataset[i]
      == dm.rx_pos_all[dm.valid_idxs_all[gidx]][:2]

  Check 4 — Spatial ordering
      mean_dist(anchor, positives) < mean_dist(anchor, negatives)
      for each sampled anchor (in 2-D Euclidean space)

  Check 5 — CSI ordering
      mean_dist(anchor, positives) < mean_dist(anchor, negatives)
      for each sampled anchor (L2 on csi_to_realvec features — phase-invariant)

  Check 6 — Shared dataset consistency  (per edge in edge_set)
      shared_pos[i, :2]    == local_dataset[bs_1].rx_pos[gidx]
      shared_pos[i, :2]    == local_dataset[bs_2].rx_pos[gidx]
      channels_bs_1[i]     == local_dataset[bs_1].channels[gidx]
      channels_bs_2[i]     == local_dataset[bs_2].channels[gidx]

Usage
-----
    python scripts/verify_dataset.py [--n-samples 20] [--bs-id 0] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use('Agg')  # headless backend — works with and without a display
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from matplotlib.patches import Circle

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, '.')

from src.datamodules.deepmimo import DeepMimoDataModule, TrajectoryCSIDataset
from src.datamodules.utils import csi_to_realvec_ferrand as csi_to_realvec

# ── helpers ───────────────────────────────────────────────────────────────────


def _ok(label: str) -> None:
    print(f'  [PASS] {label}')


def _fail(label: str, details: str) -> None:
    print(f'  [FAIL] {label}')
    print(f'         {details}')


def _check_allclose(
    label: str,
    a: np.ndarray,
    b: np.ndarray,
    # atol: float = 1e-6,
) -> bool:
    if np.all(a == b):
        _ok(label)
        return True
    max_err = float(np.max(np.abs(a - b)))
    _fail(label, f'max|a-b| = {max_err:.3e}  (shapes {a.shape} vs {b.shape})')
    return False


# ── verification routines ─────────────────────────────────────────────────────


def check_position_mapping(dm: DeepMimoDataModule, gidxs: np.ndarray) -> int:
    """Check 1: rx_pos_all_masked[gidx] == rx_pos_all[valid_idxs_all[gidx]][:2]"""
    print('\n── Check 1: position mapping ───────────────────────────────────────')
    failures = 0
    for gidx in gidxs:
        orig_idx = dm.valid_idxs_all[gidx]
        from_masked = dm.rx_pos_all_masked[gidx]  # shape (2,)
        from_original = dm.rx_pos_all[orig_idx, :2]  # shape (2,)
        label = f'gidx={gidx}  orig_idx={orig_idx}'
        if not _check_allclose(label, from_masked, from_original):
            failures += 1
    print(f'  → {len(gidxs) - failures}/{len(gidxs)} passed')
    return failures


def check_channel_mapping(dm: DeepMimoDataModule, bs_id: int, gidxs: np.ndarray) -> int:
    """Check 2: bs_coords[bs_id]['channels'][gidx] == ds[bs_id].channels[valid_idxs_all[gidx]]"""
    print(f'\n── Check 2: channel mapping  (BS {bs_id}) ──────────────────────────')
    failures = 0
    bs_channels_masked = dm.bs_coords[bs_id]['channels']  # shape (N_masked, R, T, F)
    original_channels = dm.ds[bs_id].channels  # shape (N_full, R, T, F)
    for gidx in gidxs:
        orig_idx = dm.valid_idxs_all[gidx]
        from_masked = bs_channels_masked[gidx]
        from_original = original_channels[orig_idx]
        label = f'gidx={gidx}  orig_idx={orig_idx}'
        if not _check_allclose(label, from_masked, from_original):
            failures += 1
    print(f'  → {len(gidxs) - failures}/{len(gidxs)} passed')
    return failures


def check_dataset_getitem(
    dm: DeepMimoDataModule,
    bs_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Check 3: dataset[i] with return_raw=True is consistent with original arrays."""
    print(f'\n── Check 3: dataset __getitem__ round-trip  (BS {bs_id}) ───────────')

    # Build a raw dataset from the training split for this BS
    ds_local = dm.train_local_dataset[bs_id]
    raw_ds = TrajectoryCSIDataset(
        idx_to_neg_pos=ds_local.idx_to_neg_pos,
        valid_idxs=ds_local.valid_idxs,
        rx_pos=ds_local.rx_pos,
        channels=ds_local.channels,
        pair_mode=ds_local.pair_mode,
        return_raw=True,
    )

    n = min(n_samples, len(raw_ds))
    sample_indices = rng.choice(len(raw_ds), size=n, replace=False)

    original_channels = dm.ds[bs_id].channels
    failures = 0

    for i in sample_indices:
        H_A, gidx, pos = raw_ds[int(i)]

        orig_idx = dm.valid_idxs_all[gidx]

        # CSI check
        from_original_ch = original_channels[orig_idx]
        label_ch = f'dataset[{i}]  gidx={gidx}  orig_idx={orig_idx}  CSI'
        ok_ch = _check_allclose(label_ch, H_A.numpy(), from_original_ch)

        # Position check
        from_original_pos = dm.rx_pos_all[orig_idx, :2]
        label_pos = f'dataset[{i}]  gidx={gidx}  orig_idx={orig_idx}  pos'
        ok_pos = _check_allclose(label_pos, pos.numpy(), from_original_pos)

        if not (ok_ch and ok_pos):
            failures += 1

    print(f'  → {n - failures}/{n} passed')
    return failures


def _sample_triplet_entries(
    dm: DeepMimoDataModule,
    bs_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Return a list of dicts with anchor/pos/neg gidxs for sampled entries."""
    ds_local = dm.train_local_dataset[bs_id]
    keys = list(ds_local.idx_to_neg_pos.keys())
    n = min(n_samples, len(keys))
    chosen = rng.choice(len(keys), size=n, replace=False)
    entries = []
    for i in chosen:
        key = keys[int(i)]
        anchor_gidx = key[1]
        pos_gidxs = ds_local.idx_to_neg_pos[key]['pos']
        neg_gidxs = ds_local.idx_to_neg_pos[key]['neg']
        entries.append({'anchor': anchor_gidx, 'pos': pos_gidxs, 'neg': neg_gidxs})
    return entries


def check_spatial_ordering(
    dm: DeepMimoDataModule,
    bs_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Check 4: positives are spatially closer to anchor than negatives."""
    print(f'\n── Check 4: spatial ordering  (BS {bs_id}) ────────────────────────')
    entries = _sample_triplet_entries(dm, bs_id, n_samples, rng)
    failures = 0
    for e in entries:
        anchor_pos = dm.rx_pos_all_masked[e['anchor']]  # (2,)
        pos_pos = dm.rx_pos_all_masked[e['pos']]  # (n_pos, 2)
        neg_pos = dm.rx_pos_all_masked[e['neg']]  # (n_neg, 2)

        mean_pos_dist = float(np.mean(np.linalg.norm(pos_pos - anchor_pos, axis=1)))
        mean_neg_dist = float(np.mean(np.linalg.norm(neg_pos - anchor_pos, axis=1)))

        label = (
            f'anchor={e["anchor"]}  '
            f'mean_pos_dist={mean_pos_dist:.2f}  mean_neg_dist={mean_neg_dist:.2f}'
        )
        if mean_pos_dist < mean_neg_dist:
            _ok(label)
        else:
            _fail(label, 'positives are NOT closer than negatives in position space')
            failures += 1

    print(f'  → {len(entries) - failures}/{len(entries)} passed')
    return failures


def check_csi_ordering(
    dm: DeepMimoDataModule,
    bs_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Check 5: positives have closer CSI to anchor than negatives.

    Distance is computed in the csi_to_realvec feature space (2D-FFT +
    frequency autocorrelation + log scaling), which is phase-invariant and
    matches the actual representation used during training.
    Raw complex L2 distance would be meaningless here because the carrier
    phase offset is arbitrary across snapshots.
    """
    print(f'\n── Check 5: CSI ordering  (BS {bs_id}) ────────────────────────────')
    entries = _sample_triplet_entries(dm, bs_id, n_samples, rng)
    channels = dm.bs_coords[bs_id]['channels']  # (N_masked, R, T, F) complex

    def _feat(gidx: int) -> np.ndarray:
        H = torch.from_numpy(channels[gidx])
        return csi_to_realvec(H).numpy()

    def _feat_dist(f_a: np.ndarray, f_b: np.ndarray) -> float:
        return float(np.linalg.norm(f_a - f_b))

    failures = 0
    for e in entries:
        f_anchor = _feat(e['anchor'])

        mean_pos_dist = float(np.mean([_feat_dist(f_anchor, _feat(i)) for i in e['pos']]))
        mean_neg_dist = float(np.mean([_feat_dist(f_anchor, _feat(i)) for i in e['neg']]))

        label = (
            f'anchor={e["anchor"]}  '
            f'mean_pos_dist={mean_pos_dist:.4f}  mean_neg_dist={mean_neg_dist:.4f}'
        )
        if mean_pos_dist < mean_neg_dist:
            _ok(label)
        else:
            _fail(label, 'positives are NOT closer than negatives in feature space')
            failures += 1

    print(f'  → {len(entries) - failures}/{len(entries)} passed')
    return failures


def check_shared_dataset_consistency(
    dm: DeepMimoDataModule,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Check 6: shared dataset positions and CSI match the private local datasets.

    For each sampled shared point at row i (with masked global index gidx):
      6a — shared_pos[i, :2]    == local_dataset[bs_1].rx_pos[gidx]
           shared_pos[i, :2]    == local_dataset[bs_2].rx_pos[gidx]
           (same position served by both private datasets)
      6b — channels_bs_1[i]     == local_dataset[bs_1].channels[gidx]
           (CSI stored in shared dataset matches what BS1's private dataset serves)
      6c — channels_bs_2[i]     == local_dataset[bs_2].channels[gidx]
           (CSI stored in shared dataset matches what BS2's private dataset serves)
    """
    print('\n── Check 6: shared dataset consistency ─────────────────────────────')

    if not dm.train_shared_dataset:
        print(
            '  (no shared datasets produced — either edge_set is empty or no trajectory '
            'anchors fall in the intersection of any configured BS pair coverage areas)'
        )
        return 0

    total_failures = 0

    for (bs_1, bs_2), shared_ds in dm.train_shared_dataset.items():
        print(f'  Edge BS{bs_1}↔BS{bs_2}  ({len(shared_ds)} shared points)')

        if shared_ds.shared_traj_idxs is None:
            print('  [SKIP] shared_traj_idxs not stored on this dataset')
            continue

        local_ds_1 = dm.train_local_dataset[bs_1]
        local_ds_2 = dm.train_local_dataset[bs_2]

        n = min(n_samples, len(shared_ds))
        sample_indices = rng.choice(len(shared_ds), size=n, replace=False)
        failures = 0

        for i in sample_indices:
            gidx = int(shared_ds.shared_traj_idxs[i])
            shared_pos_2d = shared_ds.shared_pos[i, :2]

            # 6a — position: both private datasets serve the same 2D position as stored
            ok_pos1 = _check_allclose(
                f'shared[{i}] gidx={gidx}  pos via BS{bs_1} private',
                shared_pos_2d,
                local_ds_1.rx_pos[gidx],
            )
            ok_pos2 = _check_allclose(
                f'shared[{i}] gidx={gidx}  pos via BS{bs_2} private',
                shared_pos_2d,
                local_ds_2.rx_pos[gidx],
            )

            # 6b — CSI BS1: shared channels_bs_1 matches what BS1's private dataset holds
            ok_ch1 = _check_allclose(
                f'shared[{i}] gidx={gidx}  CSI via BS{bs_1} private',
                shared_ds.channels_bs_1[i],
                local_ds_1.channels[gidx],
            )

            # 6c — CSI BS2: shared channels_bs_2 matches what BS2's private dataset holds
            ok_ch2 = _check_allclose(
                f'shared[{i}] gidx={gidx}  CSI via BS{bs_2} private',
                shared_ds.channels_bs_2[i],
                local_ds_2.channels[gidx],
            )

            if not (ok_pos1 and ok_pos2 and ok_ch1 and ok_ch2):
                failures += 1

        print(f'  → {n - failures}/{n} passed')
        total_failures += failures

    return total_failures


# ── triplet visualisation ─────────────────────────────────────────────────────


def plot_triplet_batch(
    dm: DeepMimoDataModule,
    bs_id: int,
    rng: np.random.Generator,
    n_anchors: int = 4,
    output_path: str = 'triplets.png',
) -> None:
    """Plot a 2×2 grid of anchor/positive/negative triplets for one BS.

    For each of the *n_anchors* randomly chosen anchors the panel shows:
      - the **full trajectory** of the owning user as a grey line
      - **anchor** as a large black star
      - **positive** samples as green circles
      - **negative** samples as red crosses

    Parameters
    ----------
    dm : DeepMimoDataModule
    bs_id : int
        Index of the base station whose local dataset is used.
    rng : np.random.Generator
    n_anchors : int
        Number of anchors to display (must be a perfect square for the grid,
        but the grid is always 2×2 so pass 4).
    output_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    ds_local = dm.train_local_dataset[bs_id]
    rx_pos = dm.rx_pos_all_masked  # (N_masked, 2)

    bs_xy = dm.ds[bs_id].bs_pos.squeeze()[:2]  # (2,)
    coverage_radius = dm.bs_coords[bs_id]['coverage_radius']

    # Group all gidxs by user so we can draw full trajectories
    user_to_gidxs: dict[int, list[int]] = {}
    for uid, gidx in ds_local.idx_to_neg_pos:
        user_to_gidxs.setdefault(uid, []).append(gidx)

    keys = list(ds_local.idx_to_neg_pos.keys())
    chosen_keys = [keys[int(i)] for i in rng.choice(len(keys), size=n_anchors, replace=False)]

    ncols = 2
    nrows = (n_anchors + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 9))
    axes = np.array(axes).reshape(-1)

    for ax, (uid, anchor_gidx) in zip(axes, chosen_keys):
        entry = ds_local.idx_to_neg_pos[(uid, anchor_gidx)]
        pos_gidxs = entry['pos']
        neg_gidxs = entry['neg']

        # Full trajectory for this user — preserve insertion order (= temporal order)
        traj_gidxs = user_to_gidxs[uid]
        traj_xy = rx_pos[traj_gidxs]  # (T, 2)

        anchor_xy = rx_pos[anchor_gidx]  # (2,)
        pos_xy = rx_pos[pos_gidxs]  # (n_pos, 2)
        neg_xy = rx_pos[neg_gidxs]  # (n_neg, 2)

        # Trajectory line
        ax.plot(
            traj_xy[:, 0],
            traj_xy[:, 1],
            color='lightgrey',
            linewidth=1.0,
            zorder=1,
            label='trajectory',
        )

        # Negative samples
        ax.scatter(
            neg_xy[:, 0],
            neg_xy[:, 1],
            c='red',
            marker='x',
            s=50,
            linewidths=1.5,
            zorder=3,
            label=f'negatives ({len(neg_gidxs)})',
        )

        # Positive samples
        ax.scatter(
            pos_xy[:, 0],
            pos_xy[:, 1],
            c='limegreen',
            marker='o',
            s=40,
            edgecolors='darkgreen',
            linewidths=0.8,
            zorder=4,
            label=f'positives ({len(pos_gidxs)})',
        )

        # Anchor
        ax.scatter(
            [anchor_xy[0]], [anchor_xy[1]], c='black', marker='*', s=200, zorder=5, label='anchor'
        )

        ax.set_title(f'BS {bs_id}  |  user {uid}  |  gidx {anchor_gidx}', fontsize=9)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')

        # Fix limits from trajectory data before adding the (potentially large) circle
        ax.autoscale_view()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # BS position and coverage area — clipped to current view
        ax.scatter(
            [bs_xy[0]],
            [bs_xy[1]],
            c='royalblue',
            marker='^',
            s=120,
            zorder=6,
            label=f'BS {bs_id}',
            clip_on=True,
        )
        ax.add_patch(
            Circle(
                (bs_xy[0], bs_xy[1]),
                coverage_radius,
                facecolor='royalblue',
                alpha=0.06,
                edgecolor='royalblue',
                linewidth=0.8,
                zorder=0,
                clip_on=True,
            )
        )

        # Restore data-driven limits so the circle doesn't rescale the axes
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Hide any unused axes (if n_anchors < nrows*ncols)
    for ax in axes[n_anchors:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Triplet sampling sanity-check — BS {bs_id}\n'
        'grey line: full user trajectory | ★: anchor | ●: positives | ✕: negatives',
        fontsize=10,
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\n  [PLOT] saved to {output_path}')
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify CSI / position index consistency')
    parser.add_argument(
        '--n-samples', type=int, default=20, help='Number of random indices to probe (default: 20)'
    )
    parser.add_argument(
        '--bs-id', type=int, default=0, help='Base-station index to verify (default: 0)'
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--plot', action='store_true', help='Generate triplet visualisation plot')
    parser.add_argument(
        '--plot-output',
        type=str,
        default=None,
        help='Save plot to this path instead of showing it (e.g. triplets.png)',
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── minimal datamodule config ─────────────────────────────────────────────
    cfg = omegaconf.OmegaConf.create(
        {
            'scenario': 'city_3_houston_3p5',
            'download': True,
            'batch_size': 4,
            'num_workers': 0,
            'pin_memory': False,
            'train_num_users': 50,
            'test_num_users': 1,
            'val_num_users': 1,
            'T_min': 1000,
            'T_max': 1001,
            'coverage_area': 0.6,
            'trajectory_kind': 'neighbor_linear',
            'pair_mode': 'triplet',
            'in_window': 5,
            'out_window': 10,
            'bias_sampling': False,
            'n_pos': None,
            'n_neg': None,
            'compute_channels': {
                'max_subcarriers': 16,
                'ue_antenna': {'shape': [1, 1]},
            },
            'edge_set': [[0, 1], [0, 2], [1, 2]],
        }
    )

    print('Setting up DeepMimoDataModule …')
    dm = DeepMimoDataModule(dataset_cfg=cfg, seed=args.seed)
    dm.prepare_data()
    dm.setup()
    print(f'  valid_idxs_all : {len(dm.valid_idxs_all)} points (union of all BS coverage areas)')
    print(f'  rx_pos_all     : {dm.rx_pos_all.shape}')
    for bs_id in dm.bs_coords:
        print(f'  BS {bs_id} channels : {dm.bs_coords[bs_id]["channels"].shape}')

    # Probe indices from valid_idxs_all
    n = min(args.n_samples, len(dm.valid_idxs_all))
    gidxs = rng.choice(len(dm.valid_idxs_all), size=n, replace=False)

    total_failures = 0
    total_failures += check_position_mapping(dm, gidxs)
    total_failures += check_channel_mapping(dm, args.bs_id, gidxs)
    total_failures += check_dataset_getitem(dm, args.bs_id, args.n_samples, rng)
    total_failures += check_spatial_ordering(dm, args.bs_id, args.n_samples, rng)
    total_failures += check_csi_ordering(dm, args.bs_id, args.n_samples, rng)
    total_failures += check_shared_dataset_consistency(dm, args.n_samples, rng)

    print('\n' + '═' * 60)
    if total_failures == 0:
        print('ALL CHECKS PASSED — indexing is consistent.')
    else:
        print(f'{total_failures} check(s) FAILED — indexing has issues.')

    if args.plot:
        print('\nGenerating triplet visualisation …')
        out = args.plot_output or f'triplets_bs{args.bs_id}.png'
        plot_triplet_batch(dm, args.bs_id, rng, n_anchors=4, output_path=out)

    if total_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
