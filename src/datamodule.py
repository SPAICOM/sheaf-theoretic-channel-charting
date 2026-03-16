"""
Lightning DataModule for generating trajectory-based
CSI datasets using DeepMIMO.

This module:
1. Downloads/loads a DeepMIMO scenario.
2. Computes channel matrices.
3. Creates synthetic user trajectories from the available receiver positions.
4. Builds train/validation/test datasets using TrajectoryCSIDataset.
5. Provides PyTorch DataLoaders for training pipelines.

The DataModule follows the PyTorch Lightning lifecycle:
    prepare_data() -> setup() -> train/val/test_dataloader()
"""

from typing import Any

import deepmimo as dm
import lightning as L
import numpy as np

# from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .dataset import TrajectoryCSIDataset


def _merge_defaults(
    defaults: dict[str, Any],
    override: Any,
) -> dict[str, Any]:
    """
    Merge default configuration values with user-provided overrides.

    This utility ensures that:
    - If the user passes an OmegaConf DictConfig,
      it is converted to a standard dict.
    - Missing parameters fall back to the provided defaults.

    Parameters
    ----------
    defaults : dict[str, Any]
        Default configuration dictionary.
    override : Any
        User-provided configuration. Can be:
        - dict
        - OmegaConf DictConfig
        - None

    Returns
    -------
    dict[str, Any]
        A merged configuration dictionary where `override`
        values take precedence over defaults.
    """
    # Convert OmegaConf configuration into a standard Python dictionary
    if isinstance(override, DictConfig):
        override = OmegaConf.to_container(override, resolve=True)

    # If override is None, treat it as an empty dictionary
    override = override or {}

    # Create a copy of defaults and update with overrides
    out = dict(defaults)
    out.update(override)

    return out


class CSIDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for trajectory-based CSI learning.

    This module wraps the DeepMIMO dataset and converts it into
    trajectory-based samples suitable for contrastive or triplet
    learning tasks on channel state information (CSI).

    Responsibilities:
    ----------------
    - Download/load a DeepMIMO scenario
    - Compute channel matrices
    - Generate trajectory datasets
    - Split datasets into train / validation / test
    - Provide DataLoaders for training pipelines

    The resulting datasets produce CSI samples arranged as:
        real/imaginary concatenated channel vectors.

    Attributes
    ----------
    cfg : dict
        Final merged configuration.
    train_dataset : TrajectoryCSIDataset | None
        Training dataset instance.
    test_dataset : TrajectoryCSIDataset | None
        Test dataset instance.
    val_dataset : TrajectoryCSIDataset | None
        Validation dataset instance.
    feature_dim : int
        Dimension of flattened CSI feature vectors (real + imaginary parts).
    """

    # Default configuration values used if the user does not specify them
    DEFAULTS: dict[str, Any] = {
        'scenario': 'asu_campus_3p5',  # DeepMIMO scenario name
        'download': True,  # whether to download the dataset if missing
        'batch_size': 64,
        'num_workers': 0,
        'shuffle': True,
        'pin_memory': True,
        'compute_channels': {},
        'num_users': 200,
        'T_min': 20,
        'T_max': 60,
        'pair_mode': 'triplet',  # "triplet" or "contrastive"
        'window': 3,
        'include_same_user_outside_window': False,
        'p_positive': 0.5,  # for contrastive
        'train_seed': 27,
        'test_seed': 42,
        'val_seed': 123,
        'train_split': 0.8,
        'val_split': 0.2,
    }

    def __init__(
        self,
        dataset_cfg: DictConfig | dict[str, Any],
    ):
        """
        Initialize the CSIDataModule.

        Parameters
        ----------
        dataset_cfg : DictConfig | dict[str, Any]
            User configuration overriding DEFAULTS.
            Can be provided via Hydra/OmegaConf or as a normal dictionary.
        """
        super().__init__()

        # Merge user configuration with defaults
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)

        # Dataset placeholders (initialized during setup)
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.n_agents = None

        # Basic validation of split parameters
        assert 0 <= self.cfg['train_split'] <= 1, (
            'The "train_split" must be between 0 and 1.'
        )
        assert 0 <= self.cfg['val_split'] <= 1, (
            'The "val_split" must be between 0 and 1.'
        )

    def prepare_data(self) -> None:
        """
        Download the DeepMIMO scenario if required.

        This method is executed **once per node** in distributed setups
        and should only contain operations that are safe to run once,
        such as dataset downloads.

        Returns
        -------
        None
        """
        # Only download if explicitly requested and supported
        if self.cfg['download'] and hasattr(dm, 'download'):
            dm.download(self.cfg['scenario'])

        return None

    # ----------------- Snapping helpers -----------------
    def _snap(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        Snap 2D coordinates to the nearest valid RX index.

        Parameters
        ----------
        xy : np.ndarray
            XY coordinates of shape (2,) or (N,2).

        Returns
        -------
        np.ndarray
            Indices of nearest valid RX positions.
        """
        _, idx_local = self.kdtree.query(xy, k=1)
        return self.valid_idxs[idx_local].astype(np.int64)

    # ----------------- Trajectory generators -----------------
    def _rand_anchor_xy(self) -> np.ndarray:
        """
        Pick a random RX XY coordinate to serve as a trajectory anchor.

        Returns
        -------
        np.ndarray
            Selected XY coordinate (2,).
        """
        if self.bias_sampling:
            power = np.linalg.norm(
                self.H_users.reshape(len(self.H_users), -1), axis=1
            )
            prob = power / power.sum()
            idx = self.rng.choice(len(self.rx_pos_all_masked), p=prob)
        else:
            idx = int(self.rng.integers(0, len(self.rx_pos_all_masked)))
        return self.rx_pos_all_masked[idx].copy()

    def _generate_one(
        self,
        kind: str | None,
        T: int,
    ) -> np.ndarray:
        """
        Generate a trajectory of length T of the specified kind.

        Parameters
        ----------
        kind : str
            One of 'linear', 'circular', or 'random'.
        T : int
            Trajectory length.

        Returns
        -------
        np.ndarray
            Array of RX indices representing the trajectory.
        """
        if kind == 'linear':
            start = self._rand_anchor_xy()
            L = float(self.rng.uniform(*self.linear_len))
            ang = float(self.rng.uniform(0, 2 * np.pi))
            end = start + L * np.array([np.cos(ang), np.sin(ang)])
            s = np.linspace(0.0, 1.0, T)
            xy = start * (1 - s)[:, None] + end * s[:, None]

        if kind == 'circular':
            center = self._rand_anchor_xy()
            r = float(self.rng.uniform(*self.circle_r))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            ang = np.linspace(0.0, 2 * np.pi, T, endpoint=False) + phase
            xy = np.stack(
                [center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)],
                axis=1,
            )

        if kind == 'random':
            xy = np.empty((T, 2), dtype=np.float64)
            xy[0] = self._rand_anchor_xy()
            prev_dir = None
            for t in range(1, T):
                step = float(self.rng.uniform(*self.random_step))
                if (
                    prev_dir is None
                    or self.rng.random() > self.random_keep_dir
                ):
                    ang = float(self.rng.uniform(0, 2 * np.pi))
                else:
                    ang = float(
                        np.arctan2(prev_dir[1], prev_dir[0])
                        + self.rng.normal(0, 0.5)
                    )
                d = np.array([np.cos(ang), np.sin(ang)])
                xy[t] = self._snap(xy[t - 1] + step * d)
                prev_dir = d
            return xy

        return self._snap(xy)

    # ----------------- CSI helpers -----------------
    def _H_from_global_index(
        self,
        gidx: int,
    ) -> torch.Tensor:
        """
        Return the CSI tensor for the given global index.

        Parameters
        ----------
        gidx : int
            Global index into the flattened trajectory dataset.

        Returns
        -------
        torch.Tensor
            Complex CSI tensor for the corresponding RX location.
        """
        return torch.from_numpy(self.H_users[gidx])  # complex tensor

    def _pick_one(self, idxs: np.ndarray) -> int:
        """
        Randomly select one index from a list of indices.

        Parameters
        ----------
        idxs : np.ndarray
            Array of candidate indices.

        Returns
        -------
        int
            Randomly selected index, or -1 if input is empty.
        """
        if idxs is None or len(idxs) == 0:
            return -1
        return int(idxs[int(self.rng.integers(0, len(idxs)))])

    def _shared_gen(self, num_users) -> None:
        for user_id in range(num_users):
            # Random trajectory length
            T = int(self.rng.integers(self.T_min, self.T_max + 1))

            # Randomly pick trajectory kind
            kind = (
                self.kinds[int(self.rng.integers(0, len(self.kinds)))]
                if self.trajectory_kind is None
                else self.trajectory_kind
            )

            # Generate trajectory of RX indices
            rx_idxs = self._generate_one(kind, T)
            max_idx = np.max(rx_idxs)
            min_idx = np.min(rx_idxs)
            for idx in rx_idxs:
                min_point = min_idx + self.out_window
                max_point = max_idx - self.out_window
                if idx > min_point and idx < max_point:
                    pos = np.clip(
                        np.arange(
                            idx - self.in_window, idx + self.in_window + 1
                        ),
                        a_min=0,
                        a_max=max_idx,
                    )
                    pos = pos[pos != idx]
                    neg = np.clip(
                        np.concat(
                            [
                                np.arange(
                                    idx - self.out_window, idx - self.in_window
                                ),
                                np.arange(
                                    idx + self.in_window + 1,
                                    idx + self.out_window + 1,
                                ),
                            ]
                        ),
                        a_min=0,
                        a_max=max_idx,
                    )
                    neg = neg[neg != idx]
                    self.idx_to_neg_pos[(user_id, idx)] = {
                        'pos': pos,
                        'neg': neg,
                    }

        local_datasets = {}
        shared_datasets = {}

        for base_station, _ in enumerate(self.ds):
            local_datasets[base_station] = TrajectoryCSIDataset(
                idx_to_neg_pos=self.idx_to_neg_pos,
                mask=self.bs_coords[base_station]['mask'],
                rx_pos=self.bs_coords[base_station]['rx_pos'],
                H_users=self.bs_coords[base_station]['channels'],
            )

        for bs_1, bs_2 in self.edge_set:
            shared_mask = (
                self.bs_coords[bs_1]['mask'] & self.bs_coords[bs_2]['mask']
            )
            shared_pos = self.rx_pos_all[np.where(shared_mask)[0]]
            channels_bs_1 = self.ds[bs_1].channels[shared_mask]
            channels_bs_2 = self.ds[bs_2].channels[shared_mask]

            shared_datasets[(bs_1, bs_2)] = SharedTrajectoryCSIDataset(
                idx_bs_1=bs_1,
                idx_bs_2=bs_2,
                shared_mask=shared_mask,
                shared_pos=shared_pos,
                channels_bs_1=channels_bs_1,
                channels_bs_2=channels_bs_2,
            )

        return local_datasets, shared_datasets

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """
        Load the DeepMIMO dataset and construct trajectory datasets.

        This method:
        1. Loads the DeepMIMO scenario
        2. Computes channel matrices
        3. Determines CSI feature dimensionality
        4. Splits the dataset into train/test/validation sets
        5. Creates TrajectoryCSIDataset instances

        Parameters
        ----------
        stage : str | None
            Stage of the Lightning lifecycle ("fit", "test", etc.).
            Currently not used but kept for Lightning compatibility.

        Returns
        -------
        None
        """

        # Load DeepMIMO scenario
        self.ds = dm.load(self.cfg['scenario'])

        # Channel computation arguments
        ch_kwargs = self.cfg.get('compute_channels') or {}

        self.n_agents = len(self.ds) if isinstance(ds, (list, tuple)) else 1

        self.rx_pos_all = (
            self.ds[0].rx_pos if isinstance(self.ds, (list, tuple)) else ds.rx_pos
        )
        self.valid_rx_pos = {}
        union_mask = np.zeros_like(self.rx_pos_all, dtype=bool)

        for base_station in self.ds:
            base_station.compute_channels(**ch_kwargs)
            bs_pos = base_station.bs_pos

            d = np.linalg.norm(self.rx_pos_all - bs_pos, axis=1)
            if self.cfg.r_min is not None:
                mask &= d >= float(self.cfg.r_min)

            assert (
                self.cfg.coverage_area >= 0 and self.cfg.coverage_area <= 1
            ), '"coverage_area" must be between 0 and 1 for a BS.'
            if self.cfg.r_max is None:
                xmin, ymin = self.rx_pos_all[:, :2].min(axis=0)
                xmax, ymax = self.rx_pos_all[:, :2].max(axis=0)
                r_max = self.cfg.coverage_area * min(xmax - xmin, ymax - ymin)

            mask &= d <= float(r_max)

            self.bs_coords[base_station] = {
                'rx_pos': self.rx_pos_all[np.where(mask)[0]],
                'mask': mask,
                'channels': base_station.channels[np.where(mask)[0]],
            }
            union_mask |= mask

        self.valid_idxs = np.where(union_mask)[0]
        self.rx_pos_all_masked = self.rx_pos_all[self.valid_idxs]
        self.rx_pos_all_masked = self.rx_pos_all_masked[:, :2]
        self.kdtree = KDTree(self.rx_pos_all_masked)

        # Compute feature dimensionality
        # Each channel is complex -> we convert to real/imag pairs
        per_sample_complex = int(np.prod(ch.shape[1:]))
        self.feature_dim = 2 * per_sample_complex

        # ---------------------------------------------------------------
        #                Compute dataset split sizes
        # ---------------------------------------------------------------

        train_num_users = int(
            int(self.cfg['train_num_users'])
        )
        test_num_users = int(
            int(self.cfg['test_num_users'])
        )
        val_num_users = int(
            int(self.cfg['val_num_users'])
        )

        # ---------------------------------------------------------------
        #                   Create the Datasets
        # ---------------------------------------------------------------

        # Training dataset
        self.train_local_dataset, self.train_shared_dataset = self._shared_gen(
            train_num_users
        )

        # Test dataset
        self.test_local_dataset, self.test_shared_dataset = self._shared_gen(
            test_num_users
        )

        # Validation dataset
        self.val_local_dataset, self.val_shared_dataset = self._shared_gen(
            val_num_users
        )

        return None

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader used during model training.
        """

        loaders = {}
        for base_station in self.train_local_dataset.keys():
            loaders[base_station] = DataLoader(
                self.train_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        for bs_1, bs_2 in self.train_shared_dataset.keys():
            loaders[(bs_1, bs_2)] = DataLoader(
                self.train_shared_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        return CombinedLoader(loaders, mode='max_size_cycle')

    def test_dataloader(self) -> DataLoader:
        """
        Create the test DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during testing/evaluation.
        """
        loaders = {}
        for base_station in self.test_local_dataset.keys():
            loaders[base_station] = DataLoader(
                self.test_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        for bs_1, bs_2 in self.test_shared_dataset.keys():
            loaders[(bs_1, bs_2)] = DataLoader(
                self.test_shared_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during validation.
        """
        loaders = {}
        for base_station in self.val_local_dataset.keys():
            loaders[base_station] = DataLoader(
                self.val_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        for bs_1, bs_2 in self.val_shared_dataset.keys():
            loaders[(bs_1, bs_2)] = DataLoader(
                self.val_shared_dataset[base_station],
                batch_size=self.cfg['batch_size'],
            )

        return CombinedLoader(loaders, mode='max_size_cycle')
