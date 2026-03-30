"""
Visualization utilities for channel charting datasets.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.datamodules.dichasus import DICHASUSDataModule


def plot_positions_2d(
    dm: 'DICHASUSDataModule',
    split: str = 'train',
    per_bs: bool = False,
    color_by_time: bool = False,
    max_samples: int | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot 2D positions from the DICHASUS dataset.

    Parameters
    ----------
    dm : DICHASUSDataModule
        The datamodule with loaded data.
    split : str
        Which split to plot: 'train', 'val', or 'test'.
    per_bs : bool
        If True, create separate subplots per BS.
        If False, overlay all BSs on one plot.
    color_by_time : bool
        If True, color points by timestamp to show trajectory.
        If False, use uniform color per BS.
    max_samples : int | None
        Maximum number of samples to plot per BS. If None, plot all.
        Useful for quick visualization with large datasets.
    save_path : str | None
        Path to save the figure. If None, don't save.
    show : bool
        If True, display the plot.
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    local_dataset_attr = f'{split}_local_dataset'
    if not hasattr(dm, local_dataset_attr):
        raise ValueError(f"Split '{split}' not available. Have you called dm.setup()?")

    local_dataset = getattr(dm, local_dataset_attr)
    base_stations = dm.base_stations

    if per_bs:
        n_bs = len(base_stations)
        fig, axes = plt.subplots(1, n_bs, figsize=(5 * n_bs, 5), squeeze=False)
        axes = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        axes = [ax]

    colors = plt.cm.viridis(np.linspace(0, 1, len(base_stations)))

    for idx, bs_id in enumerate(base_stations):
        ax = axes[idx] if per_bs else axes[0]

        ds = local_dataset[bs_id]
        valid_idxs = ds.valid_idxs

        # Subsample if max_samples is specified
        if max_samples is not None and len(valid_idxs) > max_samples:
            rng = np.random.default_rng()
            indices = rng.choice(len(valid_idxs), max_samples, replace=False)
            valid_idxs = valid_idxs[indices]

        pos_x = []
        pos_y = []
        times = []

        for gidx in valid_idxs:
            tfrecord_path, record_idx = ds.file_record_map[gidx]
            csi, pos, time = ds._load_record(tfrecord_path, record_idx)
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            times.append(time)

        pos_x = np.array(pos_x)
        pos_y = np.array(pos_y)
        times = np.array(times)

        if color_by_time:
            scatter = ax.scatter(
                pos_x,
                pos_y,
                c=times,
                cmap='viridis',
                s=5,
                alpha=0.6,
            )
            plt.colorbar(scatter, ax=ax, label='Timestamp (s)')
        else:
            ax.scatter(
                pos_x,
                pos_y,
                c=[colors[idx]],
                s=5,
                alpha=0.6,
                label=f'BS {bs_id}',
            )

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'BS {bs_id}' if per_bs else f'Positions - Split: {split}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if not per_bs:
            ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
