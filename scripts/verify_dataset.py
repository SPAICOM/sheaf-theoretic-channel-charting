"""
Verification script for CSI / position index consistency.

Checks that the three layers of indexing in CSIDataModule are mutually
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

import numpy as np
import omegaconf
import torch

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, '.')

from src.datamodule import CSIDataModule
from src.dataset import TrajectoryCSIDataset, csi_to_realvec


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

def check_position_mapping(dm: CSIDataModule, gidxs: np.ndarray) -> int:
    """Check 1: rx_pos_all_masked[gidx] == rx_pos_all[valid_idxs_all[gidx]][:2]"""
    print('\n── Check 1: position mapping ───────────────────────────────────────')
    failures = 0
    for gidx in gidxs:
        orig_idx = dm.valid_idxs_all[gidx]
        from_masked = dm.rx_pos_all_masked[gidx]          # shape (2,)
        from_original = dm.rx_pos_all[orig_idx, :2]       # shape (2,)
        label = f'gidx={gidx}  orig_idx={orig_idx}'
        if not _check_allclose(label, from_masked, from_original):
            failures += 1
    print(f'  → {len(gidxs) - failures}/{len(gidxs)} passed')
    return failures


def check_channel_mapping(dm: CSIDataModule, bs_id: int, gidxs: np.ndarray) -> int:
    """Check 2: bs_coords[bs_id]['channels'][gidx] == ds[bs_id].channels[valid_idxs_all[gidx]]"""
    print(f'\n── Check 2: channel mapping  (BS {bs_id}) ──────────────────────────')
    failures = 0
    bs_channels_masked = dm.bs_coords[bs_id]['channels']   # shape (N_masked, R, T, F)
    original_channels = dm.ds[bs_id].channels              # shape (N_full, R, T, F)
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
    dm: CSIDataModule,
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
    dm: CSIDataModule,
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
    dm: CSIDataModule,
    bs_id: int,
    n_samples: int,
    rng: np.random.Generator,
) -> int:
    """Check 4: positives are spatially closer to anchor than negatives."""
    print(f'\n── Check 4: spatial ordering  (BS {bs_id}) ────────────────────────')
    entries = _sample_triplet_entries(dm, bs_id, n_samples, rng)
    failures = 0
    for e in entries:
        anchor_pos = dm.rx_pos_all_masked[e['anchor']]          # (2,)
        pos_pos    = dm.rx_pos_all_masked[e['pos']]             # (n_pos, 2)
        neg_pos    = dm.rx_pos_all_masked[e['neg']]             # (n_neg, 2)

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
    dm: CSIDataModule,
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
    dm: CSIDataModule,
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Verify CSI / position index consistency')
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Number of random indices to probe (default: 20)')
    parser.add_argument('--bs-id', type=int, default=0,
                        help='Base-station index to verify (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── minimal datamodule config ─────────────────────────────────────────────
    cfg = omegaconf.OmegaConf.create({
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
    })

    print('Setting up CSIDataModule …')
    dm = CSIDataModule(dataset_cfg=cfg, seed=args.seed)
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
        sys.exit(1)


if __name__ == '__main__':
    main()
