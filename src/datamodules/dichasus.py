"""
PyTorch Lightning DataModule for the DICHASUS real-world CSI dataset.

This module provides lazy loading of DICHASUS TFRecord files using the tfrecord
library. Unlike DeepMIMO (simulated + synthetic trajectories), this module:

1. Loads real TFRecord files from the DICHASUS distributed channel sounder.
2. Applies STO/CPO calibration via per-file offset JSON files.
3. Splits the full antenna array into per-BS sub-arrays.
4. Builds positive/negative sample pairs from timestamps (time-coherence window Tc).
5. Treats each TFRecord file as one independent measurement campaign.
6. Uses lazy loading - CSI loaded only when __getitem__ is called.

Memory Efficiency
-----------------
Only timestamps (small) are loaded into memory upfront. The actual CSI data
is loaded on-demand from TFRecord files during training. This allows handling
large datasets without loading everything into RAM.

DICHASUS antenna layout (cf0x datasets)
---------------------------------------
The raw CSI tensor has shape (total_antennas, 1024) with all BSs concatenated.
For the cf0x family the default split is 4 BSs × 8 antennas each (= 32 total).
This is configurable via ``n_bs`` and ``antennas_per_bs``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as l
import numpy as np
import requests
import tfrecord
import torch
import urllib3
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import csi_to_realvec

# Suppress SSL warnings when downloading from DICHASUS website
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _merge_defaults(
    defaults: dict[str, Any],
    override: Any,
) -> dict[str, Any]:
    """Merge default configuration values with user-provided overrides.

    Parameters
    ----------
    defaults : dict
        Default configuration dictionary.
    override : dict | DictConfig | None
        User-provided configuration to merge with defaults.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    if override is None:
        return defaults.copy()
    if isinstance(override, dict):
        return {**defaults, **override}
    if isinstance(override, DictConfig):
        return {**defaults, **dict(override.items())}  # type: ignore[return-value]
    msg = f'Cannot merge defaults of type {type(override)} with override'
    raise TypeError(msg)


class LazyDICHASUSTrajectoryDataset(Dataset):
    """Lazy-loading trajectory dataset for DICHASUS CSI data.

    This dataset loads CSI data on-demand from TFRecord files rather than
    storing all data in memory. It maintains an index mapping that tracks
    which file and record index contains each sample.

    Supports two sampling modes:
    - ``'triplet'``: returns (xA, xP, xN, y=-1) for triplet loss
    - ``'contrastive'``: returns (xA, xP, xN, y in {0,1}) for contrastive loss

    CSI is loaded lazily from TFRecord files only when __getitem__ is called.

    Parameters
    ----------
    idx_to_neg_pos : dict
        Mapping from (user_id, rx_idx) to dict with 'pos' and 'neg' keys.
        These are global indices into the concatenated dataset.
    valid_idxs : np.ndarray
        Valid indexes for the specific base station (after train/val/test split).
    file_record_map : dict[int, tuple[str, int]]
        Maps global index to (tfrecord_path, record_index) tuple.
        This is the core of lazy loading - we know WHERE each sample lives
        without loading the actual data.
    timestamps : np.ndarray
        Timestamps for each global index (for pos/neg selection based on Tc).
    file_offsets : dict[str, int]
        Offset to add to record indices for each file (for global idx calculation).
    antennas_per_bs : int
        Number of antennas per base station.
    n_bs : int
        Number of base stations.
    bs_id : int
        Which BS this dataset represents (for antenna slicing).
    pair_mode : str
        'triplet' or 'contrastive' sampling mode.
    p_positive : float
        Probability of positive pair for contrastive mode.
    preprocess : str
        Preprocessing method for CSI (e.g., 'dichasus').
    offsets_path : str | None
        Path to calibration offsets JSON (per-file). If None, no calibration applied.
    """

    def __init__(
        self,
        idx_to_neg_pos: dict[tuple[int, int], dict[str, np.ndarray]],
        valid_idxs: np.ndarray,
        file_record_map: dict[int, tuple[str, int]],
        timestamps: np.ndarray,
        file_offsets: dict[str, int],
        antennas_per_bs: int,
        n_bs: int,
        bs_ids: list[int],
        pair_mode: str = 'triplet',
        p_positive: float = 0.5,
        preprocess: str = 'dichasus',
        offsets_path: str | None = None,
    ):
        super().__init__()
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.p_positive = float(p_positive)
        self.preprocess = preprocess
        self.rng = np.random.default_rng()

        # Filter idx_to_neg_pos to only include valid indices (train/val/test split)
        self.idx_to_neg_pos = idx_to_neg_pos.copy()
        self.valid_idxs = valid_idxs
        for user_id, idx in list(self.idx_to_neg_pos.keys()):
            if idx not in self.valid_idxs:
                del self.idx_to_neg_pos[(user_id, idx)]

        # Lazy loading infrastructure
        self.file_record_map = file_record_map
        self.timestamps = timestamps
        self.file_offsets = file_offsets
        self.antennas_per_bs = antennas_per_bs
        self.n_bs = n_bs
        self.bs_ids = bs_ids
        self.total_antennas = antennas_per_bs * n_bs

        # Load STO/CPO calibration offsets if available
        # These correct for Sampling Time Offset and Carrier Phase Offset
        self.offsets = None
        if offsets_path and Path(offsets_path).exists():
            with open(offsets_path) as f:
                self.offsets = json.load(f)

        # Cache for memory-mapped numpy arrays - avoids reopening files on every access
        # Key: tfrecord_path, Value: (csi_mmap, pos_mmap, time_mmap)
        self._mmap_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_mmap(self, tfrecord_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return memory-mapped arrays for a given TFRecord path, with caching."""
        if tfrecord_path not in self._mmap_cache:
            data_dir = Path(tfrecord_path).parent
            stem = Path(tfrecord_path).stem
            csi_mm = np.load(str(data_dir / f'{stem}_csi.npy'), mmap_mode='r')
            pos_mm = np.load(str(data_dir / f'{stem}_pos.npy'), mmap_mode='r')
            time_mm = np.load(str(data_dir / f'{stem}_time.npy'), mmap_mode='r')
            self._mmap_cache[tfrecord_path] = (csi_mm, pos_mm, time_mm)
        return self._mmap_cache[tfrecord_path]

    def _load_record(self, tfrecord_path: str, record_idx: int) -> tuple:
        """Load a single record via memory-mapped numpy files (O(1) random access).

        Parameters
        ----------
        tfrecord_path : str
            Path to the (original) TFRecord file — used to derive the numpy paths.
        record_idx : int
            Index of the record within the file.

        Returns
        -------
        tuple
            (csi, position, timestamp) where:
            - csi: np.ndarray of shape (total_antennas, 1, 1024), complex64
            - position: np.ndarray of shape (2,), float64
            - timestamp: float32
        """
        csi_mm, pos_mm, time_mm = self._get_mmap(tfrecord_path)
        # .copy() materialises the slice so downstream ops don't modify the mmap
        csi = csi_mm[record_idx].copy()  # (total_antennas, 1, 1024), complex64
        pos = pos_mm[record_idx].copy()  # (2,), float64
        time = float(time_mm[record_idx])
        return csi, pos, time

    def _H_from_global_index(self, gidx: int) -> torch.Tensor:
        """Load CSI for a specific global index.

        Parameters
        ----------
        gidx : int
            Global index into the concatenated dataset.

        Returns
        -------
        torch.Tensor
            CSI tensor for this BS, shape (antennas_per_bs, 1, 1024), complex64
        """
        # Look up which file and record contains this sample
        tfrecord_path, record_idx = self.file_record_map[gidx]
        csi, _, _ = self._load_record(tfrecord_path, record_idx)

        # Slice each physical BS's antennas and concatenate along antenna dim.
        # For a single-BS group this is identical to the old scalar bs_id path.
        # BS k occupies antennas [k*apb : (k+1)*apb] in the raw array.
        apb = self.antennas_per_bs
        slices = [csi[k * apb : (k + 1) * apb] for k in self.bs_ids]
        csi_bs = np.concatenate(slices, axis=0)
        return torch.from_numpy(csi_bs)

    def _pos_from_global_index(self, gidx: int) -> torch.Tensor:
        """Load position for a specific global index.

        Parameters
        ----------
        gidx : int
            Global index into the concatenated dataset.

        Returns
        -------
        torch.Tensor
            Position tensor, shape (2,), float64 (X, Y)
        """
        tfrecord_path, record_idx = self.file_record_map[gidx]
        _, pos, _ = self._load_record(tfrecord_path, record_idx)
        return torch.from_numpy(pos)

    def __len__(self) -> int:
        """Return number of anchor samples in this dataset."""
        return len(self.idx_to_neg_pos)

    def __getitem__(self, index: int):
        """Get a training sample (anchor, positive, negative triplet).

        Returns
        -------
        tuple
            For triplet mode: (xA, xP, xN, y, pos)
            - xA: Anchor CSI, shape (feature_dim,)
            - xP: Positive CSI(s), shape (n_pos, feature_dim) or (feature_dim,)
            - xN: Negative CSI(s), shape (n_neg, feature_dim) or (feature_dim,)
            - y: Label (-1 for triplet)
            - pos: Position, shape (2,)

            For contrastive mode: (xA, xP, xN, y, pos)
            - y is 1 if positive pair, 0 if negative pair
        """
        # Look up global index for this anchor
        id = list(self.idx_to_neg_pos.keys())[index]
        gidx = id[1]

        # Load anchor CSI and position (lazy - only loads when needed)
        H_A = self._H_from_global_index(gidx)
        pos = self._pos_from_global_index(gidx)

        # Get positive and negative sample indices
        pos_idxs = self.idx_to_neg_pos[id]['pos']
        neg_idxs = self.idx_to_neg_pos[id]['neg']

        # Pre-declare tensors for type checker
        xA: torch.Tensor
        xP: torch.Tensor
        xN: torch.Tensor
        y: torch.Tensor

        # Handle triplet vs contrastive modes
        match self.pair_mode:
            case 'triplet':
                # Triplet loss: anchor, positive, negative with fixed y=-1
                xA = csi_to_realvec(H_A, method=self.preprocess)
                xP = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i), method=self.preprocess)
                        for i in pos_idxs
                    ]
                )
                xN = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i), method=self.preprocess)
                        for i in neg_idxs
                    ]
                )
                y = torch.tensor(-1, dtype=torch.long)
            case 'contrastive':
                # Contrastive loss: randomly choose positive or negative as pair
                xA = csi_to_realvec(H_A, method=self.preprocess)
                xP = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i), method=self.preprocess)
                        for i in pos_idxs
                    ]
                )
                xN = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i), method=self.preprocess)
                        for i in neg_idxs
                    ]
                )
                y = torch.tensor(
                    1 if self.rng.random() < self.p_positive else 0,
                    dtype=torch.long,
                )
            case _:
                raise ValueError(f'Unknown pair_mode: {self.pair_mode}')

        return xA, xP, xN, y, pos


class LazyDICHASUSSharedDataset(Dataset):
    """Lazy-loading shared dataset for inter-BS alignment.

    This dataset provides paired CSI samples from two different base stations
    at the same UE positions. Used for training models that learn alignment
    between different BS perspectives.

    Parameters
    ----------
    shared_global_idxs : np.ndarray
        Global indices for shared samples (positions seen by both BSs).
    file_record_map : dict[int, tuple[str, int]]
        Maps global index to (tfrecord_path, record_index).
    antennas_per_bs : int
        Number of antennas per base station.
    n_bs : int
        Number of base stations.
    idx_bs_1 : int
        First base station ID (e.g., 0 for BS1).
    idx_bs_2 : int
        Second base station ID (e.g., 1 for BS2).
    preprocess : str
        Preprocessing method for CSI.
    offsets_path : str | None
        Path to calibration offsets JSON.
    """

    def __init__(
        self,
        shared_global_idxs: np.ndarray,
        file_record_map: dict[int, tuple[str, int]],
        antennas_per_bs: int,
        n_bs: int,
        bs_ids_1: list[int],
        bs_ids_2: list[int],
        preprocess: str = 'dichasus',
        offsets_path: str | None = None,
    ):
        super().__init__()
        self.shared_global_idxs = shared_global_idxs
        self.file_record_map = file_record_map
        self.antennas_per_bs = antennas_per_bs
        self.n_bs = n_bs
        self.bs_ids_1 = bs_ids_1
        self.bs_ids_2 = bs_ids_2
        self.total_antennas = antennas_per_bs * n_bs
        self.preprocess = preprocess

        # Load calibration offsets (same as in LazyDICHASUSTrajectoryDataset)
        self.offsets = None
        if offsets_path and Path(offsets_path).exists():
            with open(offsets_path) as f:
                self.offsets = json.load(f)

        # Cache for memory-mapped numpy arrays
        self._mmap_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_mmap(self, tfrecord_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return memory-mapped arrays for a given TFRecord path, with caching."""
        if tfrecord_path not in self._mmap_cache:
            data_dir = Path(tfrecord_path).parent
            stem = Path(tfrecord_path).stem
            csi_mm = np.load(str(data_dir / f'{stem}_csi.npy'), mmap_mode='r')
            pos_mm = np.load(str(data_dir / f'{stem}_pos.npy'), mmap_mode='r')
            time_mm = np.load(str(data_dir / f'{stem}_time.npy'), mmap_mode='r')
            self._mmap_cache[tfrecord_path] = (csi_mm, pos_mm, time_mm)
        return self._mmap_cache[tfrecord_path]

    def _load_record(self, tfrecord_path: str, record_idx: int) -> np.ndarray:
        """Load CSI for a single record via memory-mapped numpy files (O(1) random access).

        Parameters
        ----------
        tfrecord_path : str
            Path to the (original) TFRecord file — used to derive the numpy paths.
        record_idx : int
            Index of the record within the file.

        Returns
        -------
        np.ndarray
            CSI tensor, shape (total_antennas, 1, 1024), complex64
        """
        csi_mm, _, _ = self._get_mmap(tfrecord_path)
        return csi_mm[record_idx].copy()  # (total_antennas, 1, 1024), complex64

    def __len__(self) -> int:
        """Return number of shared samples."""
        return len(self.shared_global_idxs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Get a shared sample from two base stations.

        Parameters
        ----------
        index : int
            Index into shared_global_idxs.

        Returns
        -------
        tuple
            (H_1, H_2, (idx_bs_1, idx_bs_2))
            - H_1: Preprocessed CSI from BS1, shape (feature_dim,)
            - H_2: Preprocessed CSI from BS2, shape (feature_dim,)
            - (idx_bs_1, idx_bs_2): Tuple identifying the BS pair
        """
        # Get global index and load CSI (lazy)
        gidx = self.shared_global_idxs[index]
        tfrecord_path, record_idx = self.file_record_map[gidx]
        csi = self._load_record(tfrecord_path, record_idx)

        # Concatenate antenna slices for each physical BS group
        apb = self.antennas_per_bs
        csi_bs_1 = np.concatenate([csi[k * apb : (k + 1) * apb] for k in self.bs_ids_1], axis=0)
        csi_bs_2 = np.concatenate([csi[k * apb : (k + 1) * apb] for k in self.bs_ids_2], axis=0)

        # Preprocess and convert to real-valued vector
        H_1 = csi_to_realvec(torch.from_numpy(csi_bs_1), method=self.preprocess)
        H_2 = csi_to_realvec(torch.from_numpy(csi_bs_2), method=self.preprocess)

        return H_1, H_2, (self.bs_ids_1[0], self.bs_ids_2[0])


class DICHASUSDataModule(l.LightningDataModule):
    """PyTorch Lightning DataModule for trajectory-based CSI learning on DICHASUS data.

    This DataModule handles the DICHASUS dataset with lazy loading via tfrecord.
    It downloads data from DaRUS (data repository) and applies calibration.

    Data Flow
    ---------
    1. prepare_data(): Download TFRecord files and calibration JSONs
    2. setup(): Build index mappings and create datasets (lazy - no CSI loaded)
    3. train_dataloader(): Returns CombinedLoader with lazy datasets

    Memory Efficiency
    -----------------
    - TFRecord files are downloaded once
    - Timestamps are loaded upfront (small, ~MB for 100k samples)
    - Actual CSI is loaded on-demand during training batches
    - File-level caching avoids re-reading same file multiple times per epoch

    Parameters
    ----------
    dataset_cfg : DictConfig | dict | None
        Configuration overriding DEFAULTS.
    seed : int | None
        Global random seed for shuffling and pair selection.

    Attributes
    ----------
    n_agents : int
        Number of distributed BSs = ``n_bs``.
    feature_dim : int
        Dimension of the per-BS CSI feature vector (set during ``setup``).
    train/val/test_local_dataset : dict[int, LazyDICHASUSTrajectoryDataset]
        ``{bs_id: dataset}`` — one entry per BS.
    train/val/test_shared_dataset : dict[tuple[int,int], LazyDICHASUSSharedDataset]
        One entry per BS pair in ``edge_set``.
    train/val/test_traj_dict : dict
        Positive/negative index mapping for each split.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> from src.datamodules.dichasus import DICHASUSDataModule
    >>> cfg = OmegaConf.load('config/hydra/dataset/dichasus_cf0x.yaml')
    >>> dm = DICHASUSDataModule(cfg)
    >>> dm.prepare_data()  # Download data
    >>> dm.setup('train')  # Build datasets
    >>> train_loader = dm.train_dataloader()
    """

    # TFRecord DOIs from DaRUS ( Stuttgart Research and Trusted Exchange)
    # Each DOI corresponds to one measurement campaign/file
    ALL_FILES: dict[str, str] = {
        'dichasus-cf02': 'doi:10.18419/darus-2854/8',
        'dichasus-cf03': 'doi:10.18419/darus-2854/9',
        'dichasus-cf04': 'doi:10.18419/darus-2854/10',
        'dichasus-cf05': 'doi:10.18419/darus-2854/11',
        'dichasus-cf06': 'doi:10.18419/darus-2854/12',
        'dichasus-cf07': 'doi:10.18419/darus-2854/13',
    }

    # Base URL for calibration offset files (STO/CPO corrections)
    _OFFSETS_BASE_URL = 'https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/'

    # Default configuration
    # These can be overridden via dataset_cfg parameter or YAML config
    DEFAULTS: dict[str, Any] = {
        'data_dir': 'dichasus',
        'files': [
            'dichasus-cf02',
            'dichasus-cf03',
            'dichasus-cf04',
        ],
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        # Each inner list is one virtual BS; single-element lists = no aggregation.
        # e.g. [[0],[1],[2]] → 3 independent BSs; [[0,1],[2,3]] → 2 aggregated BSs.
        'bs_aggregation': [[0], [1], [2]],
        'antennas_per_bs': 8,
        'Tc_pos': 1.5,  # Positive time-coherence window (seconds)
        'pair_mode': 'triplet',
        'p_positive': 0.5,
        'n_pos': None,  # Max positives per anchor (None = all)
        'n_neg': None,  # Max negatives per anchor (None = all)
        'preprocess': 'dichasus',
        'edge_set': [],  # Virtual BS index pairs for shared datasets
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'train_seed': 27,
        'test_seed': 42,
        'val_seed': 123,
    }

    def __init__(
        self,
        dataset_cfg: DictConfig | dict[str, Any] | None = None,
        anchor_seed: int | None = None,
        triplet_seed: int | None = None,
    ) -> None:
        """Initialize the DICHASUS DataModule.

        Parameters
        ----------
        dataset_cfg : DictConfig | dict | None
            Configuration dictionary to override defaults.
        anchor_seed : int | None
            Seed for sample-level train/val/test splitting (ensures same anchors across folds).
            If None, uses 'seed' from dataset_cfg or defaults to train_seed.
        triplet_seed : int | None
            Seed for sampling positives and negatives (ensures different triplets per fold).
            If None, uses 'seed' from dataset_cfg or defaults to train_seed.
        """
        super().__init__()
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)

        # Handle seeds: support both new naming (anchor_seed, triplet_seed) and legacy (seed)
        if anchor_seed is None:
            anchor_seed = self.cfg.get('seed', self.cfg.get('train_seed', 27))
        if triplet_seed is None:
            triplet_seed = self.cfg.get('seed', self.cfg.get('train_seed', 27))

        self.anchor_seed = anchor_seed
        self.triplet_seed = triplet_seed

        # Separate RNGs for anchor generation vs triplet sampling
        self.anchor_rng = np.random.default_rng(self.anchor_seed)
        self.triplet_rng = np.random.default_rng(self.triplet_seed)
        self.data_dir = Path(self.cfg['data_dir'])

        # File stems to load - controls which measurement campaigns to use
        # Each stem corresponds to one TFRecord file
        self.file_stems: list[str] = self.cfg['files']

        # Each group is one virtual BS; a single-element group means no aggregation.
        self.virtual_bs_groups: list[list[int]] = [list(g) for g in self.cfg['bs_aggregation']]
        # Flat list of all referenced physical BS IDs (derived, not configured separately)
        self.base_stations: list[int] = [bs for g in self.virtual_bs_groups for bs in g]
        self.n_bs: int = len(self.base_stations)
        self.antennas_per_bs: int = int(self.cfg['antennas_per_bs'])
        self.total_antennas: int = 32  # Raw DICHASUS cf0x has 32 antennas total

        # Time-coherence parameter for positive pair selection
        self.Tc_pos = float(self.cfg['Tc_pos'])
        self.preprocess: str = self.cfg.get('preprocess', 'dichasus')

        # n_agents is the number of virtual (possibly aggregated) BSs
        self.n_agents: int = len(self.virtual_bs_groups)
        self.feature_dim: int | None = None

        # BS pairs for shared (inter-BS alignment) loss.
        # edge_set references virtual BS indices (0-indexed into virtual_bs_groups).
        raw_edge_set: list = self.cfg.get('edge_set', [])
        n_virt = len(self.virtual_bs_groups)
        self.edge_set: list[tuple[int, int]] = []
        for e in raw_edge_set:
            v1, v2 = int(e[0]), int(e[1])
            if v1 < n_virt and v2 < n_virt:
                self.edge_set.append((v1, v2))

    def prepare_data(self) -> None:
        """Download TFRecord files and calibration offsets.

        This method runs once on the main process before any workers start.
        It downloads:
        - TFRecord files from DaRUS (DOI-based URLs)
        - Calibration offset JSONs from DICHASUS website (STO/CPO corrections)

        Files are only downloaded if they don't already exist in data_dir.
        """
        # Base URL for DaRUS API
        darus_base = (
            'https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId='
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Download each selected file
        for stem in self.file_stems:
            # TFRecord file (the main CSI data)
            tfrecords_dest = self.data_dir / f'{stem}.tfrecords'
            if not tfrecords_dest.exists():
                self._download(darus_base + self.ALL_FILES[stem], tfrecords_dest)

            # Calibration offsets (STO/CPO corrections)
            offsets_dest = self.data_dir / f'reftx-offsets-{stem}.json'
            if not offsets_dest.exists():
                self._download(
                    self._OFFSETS_BASE_URL + f'reftx-offsets-{stem}.json',
                    offsets_dest,
                )

            # Convert TFRecord to numpy memmaps for fast random access
            self._convert_tfrecord_to_numpy(stem)

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Download a file with progress bar.

        Parameters
        ----------
        url : str
            URL to download from.
        dest : Path
            Destination file path.
        """
        r = requests.get(url, stream=True, verify=False)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with (
            open(dest, 'wb') as f,
            tqdm(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB chunks
                f.write(chunk)
                bar.update(len(chunk))

    def _convert_tfrecord_to_numpy(self, stem: str) -> None:
        """Convert a TFRecord file to numpy memmap files for O(1) random access.

        Produces three files in ``data_dir``:
        - ``{stem}_csi.npy``  — shape (N, 32, 1, 1024), dtype complex64
          STO/CPO calibration and fftshift applied at conversion time.
        - ``{stem}_pos.npy``  — shape (N, 2), dtype float64
        - ``{stem}_time.npy`` — shape (N,), dtype float32

        Conversion is skipped when all three files already exist.

        Parameters
        ----------
        stem : str
            File stem (e.g. 'dichasus-cf02').
        """
        csi_path = self.data_dir / f'{stem}_csi.npy'
        pos_path = self.data_dir / f'{stem}_pos.npy'
        time_path = self.data_dir / f'{stem}_time.npy'

        if csi_path.exists() and pos_path.exists() and time_path.exists():
            return

        tfrecords_path = str(self.data_dir / f'{stem}.tfrecords')
        description = {'csi': 'byte', 'pos-tachy': 'byte', 'time': 'float'}

        # Load STO/CPO calibration once (apply per-record during conversion)
        offsets_path = self.data_dir / f'reftx-offsets-{stem}.json'
        phase: np.ndarray | None = None
        if offsets_path.exists():
            with open(offsets_path) as f:
                offsets = json.load(f)
            subcarriers = 1024
            sc_idx = np.arange(subcarriers, dtype=np.float32)
            sto = np.tensordot(
                np.array(offsets['sto'], dtype=np.float32),
                2.0 * np.pi * sc_idx / subcarriers,
                axes=0,
            )
            cpo = np.tensordot(
                np.array(offsets['cpo'], dtype=np.float32),
                np.ones(subcarriers, dtype=np.float32),
                axes=0,
            )
            phase = np.exp(1j * (sto + cpo)).astype(np.complex64)  # (32, 1024)

        # Pass 1: count records so we can pre-allocate memmaps
        loader = tfrecord.tfrecord_loader(tfrecords_path, None, description)
        N = sum(1 for _ in loader)

        total_antennas = 32  # DICHASUS cf0x layout
        csi_mm = np.lib.format.open_memmap(
            str(csi_path), mode='w+', dtype=np.complex64, shape=(N, total_antennas, 1, 1024)
        )
        pos_mm = np.lib.format.open_memmap(str(pos_path), mode='w+', dtype=np.float64, shape=(N, 2))
        time_mm = np.lib.format.open_memmap(str(time_path), mode='w+', dtype=np.float32, shape=(N,))

        # Pass 2: fill memmaps — streams one record at a time (constant RAM)
        loader = tfrecord.tfrecord_loader(tfrecords_path, None, description)
        for i, record in enumerate(tqdm(loader, total=N, desc=f'Converting {stem} to numpy')):
            # --- CSI ---
            expected_bytes = total_antennas * 1024 * 2 * 4
            csi_bytes = record['csi'][1 : 1 + expected_bytes]
            csi = np.frombuffer(csi_bytes, dtype=np.float32).reshape(total_antennas, 1024, 2).copy()
            csi = (csi[:, :, 0] + 1j * csi[:, :, 1]).astype(np.complex64)
            csi = np.fft.fftshift(csi, axes=1)
            if phase is not None:
                csi = (csi * phase).astype(np.complex64)
            csi_mm[i] = csi[:, np.newaxis, :]  # (32, 1, 1024)

            # --- Position ---
            pos_bytes = record['pos-tachy']
            pos_field_start = pos_bytes.find(b'\x22\x18')
            if pos_field_start >= 0:
                pos_data = pos_bytes[pos_field_start + 2 : pos_field_start + 26]
                pos = np.frombuffer(pos_data, dtype=np.float64)[:2].copy()
            else:
                pos = np.zeros(2, dtype=np.float64)
            pos_mm[i] = pos

            # --- Timestamp ---
            time_mm[i] = float(record['time'].item())

        # Flush buffers to disk
        del csi_mm, pos_mm, time_mm

    def _parse_tfrecord_timestamps(
        self,
        stem: str,
    ) -> tuple[np.ndarray, int, dict[str, int]]:
        """Parse only timestamps from a TFRecord file (not CSI).

        This is used to build the positive/negative index mapping without
        loading all the CSI data into memory.

        Parameters
        ----------
        stem : str
            File stem (e.g., 'dichasus-cf02').

        Returns
        -------
        tuple
            (timestamps, count, file_offset) where:
            - timestamps: sorted array of timestamps
            - count: number of records
            - file_offset: dict with file offset (always 0 for single file)
        """
        tfrecords_path = str(self.data_dir / f'{stem}.tfrecords')

        # Describe the record structure for tfrecord library
        description = {
            'csi': 'byte',
            'pos-tachy': 'byte',
            'time': 'float',
        }
        loader = tfrecord.tfrecord_loader(tfrecords_path, None, description)

        # Extract only timestamps (much smaller than full CSI data)
        timestamps_arr = np.array([record['time'].item() for record in loader], dtype=np.float32)
        # Sort by timestamp (measurements may have minor reordering)
        timestamps_arr = np.sort(timestamps_arr)

        file_offset = {stem: 0}

        return timestamps_arr, len(timestamps_arr), file_offset

    def _build_idx_to_neg_pos(
        self,
        timestamps: np.ndarray,
        file_id: int,
        global_offset: int,
        rng: np.random.Generator,
    ) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        """Build positive/negative index mapping based on time-coherence.

        Uses binary search on sorted timestamps for O(N log N) complexity.

        Positive samples: |Δt| ≤ Tc_pos (samples close in time)
        Negative samples: |Δt| > Tc_pos (any sample outside the positive window,
            drawn from the full dataset — matches DICHASUS tutorial approach)

        Parameters
        ----------
        timestamps : np.ndarray
            Sorted timestamps for all samples.
        file_id : int
            File ID to use as user_id (for cross-file isolation).
        global_offset : int
            Offset to add to local indices for global indexing.
        rng : np.random.Generator
            Random generator for optional subsampling.

        Returns
        -------
        dict
            Mapping from (file_id, global_idx) to {'pos': pos_indices, 'neg': neg_indices}
        """
        N = len(timestamps)
        n_pos = self.cfg.get('n_pos')
        n_neg = self.cfg.get('n_neg')
        idx_to_neg_pos: dict = {}

        all_indices = np.arange(N, dtype=np.int64)

        for i in range(N):
            t = timestamps[i]

            # Find positive samples: within Tc_pos time window
            lo_pos = int(np.searchsorted(timestamps, t - self.Tc_pos, side='left'))
            hi_pos = int(np.searchsorted(timestamps, t + self.Tc_pos, side='right'))
            pos_local = np.arange(lo_pos, hi_pos, dtype=np.int64)
            pos_local = pos_local[pos_local != i]  # Exclude self

            # Find negative samples: any sample outside the positive window.
            # Matches the DICHASUS tutorial approach: negatives are drawn from the
            # full dataset rather than a restricted Tc_neg window, so they are
            # statistically far in space from the anchor.
            neg_local = all_indices[(all_indices < lo_pos) | (all_indices >= hi_pos)]

            # Skip if not enough samples
            if len(pos_local) == 0 or len(neg_local) == 0:
                continue
            if n_pos is not None and len(pos_local) < n_pos:
                continue
            if n_neg is not None and len(neg_local) < n_neg:
                continue

            # Random subsample if n_pos/n_neg specified
            if n_pos is not None:
                pos_local = rng.choice(pos_local, size=n_pos, replace=False)
            if n_neg is not None:
                neg_local = rng.choice(neg_local, size=n_neg, replace=False)

            # Convert to global indices
            global_i = global_offset + i
            idx_to_neg_pos[(file_id, global_i)] = {
                'pos': pos_local + global_offset,
                'neg': neg_local + global_offset,
            }

        return idx_to_neg_pos

    def _build_file_record_map(
        self,
        stems: list[str],
    ) -> tuple[dict[int, tuple[str, int]], np.ndarray, dict[str, int]]:
        """Build the lazy loading index mapping.

        This creates a mapping from global sample index to (file_path, record_index).
        This is the core of lazy loading - we know WHERE each sample is without
        loading the actual CSI data.

        Parameters
        ----------
        stems : list[str]
            List of file stems to process.

        Returns
        -------
        tuple
            (file_record_map, timestamps_all, file_offsets) where:
            - file_record_map: dict mapping global_idx -> (tfrecord_path, record_idx)
              Indices are sorted by timestamp for time-coherence calculations
            - timestamps_all: concatenated timestamps from all files (sorted)
            - file_offsets: dict mapping stem -> starting global_idx
        """
        file_record_map: dict[int, tuple[str, int]] = {}
        all_timestamps: list[np.ndarray] = []
        file_offsets: dict[str, int] = {}
        global_idx = 0

        for stem in stems:
            tfrecords_path = str(self.data_dir / f'{stem}.tfrecords')
            file_offsets[stem] = global_idx

            # Load timestamps from pre-converted numpy file (fast, no tfrecord scan)
            file_timestamps = np.load(str(self.data_dir / f'{stem}_time.npy')).ravel()
            N = len(file_timestamps)

            for record_idx in range(N):
                file_record_map[global_idx] = (tfrecords_path, record_idx)
                global_idx += 1

            all_timestamps.append(file_timestamps)

        # Concatenate all timestamps and ensure 1D
        timestamps_all = np.concatenate(all_timestamps).ravel()

        # Sort by timestamp and reorder file_record_map accordingly
        # This is critical for time-coherence calculations
        sort_idx = np.argsort(timestamps_all)
        timestamps_all = timestamps_all[sort_idx]

        # Reindex file_record_map to sorted order
        sorted_file_record_map: dict[int, tuple[str, int]] = {}
        for new_idx, old_idx in enumerate(sort_idx):
            sorted_file_record_map[new_idx] = file_record_map[old_idx]

        return sorted_file_record_map, timestamps_all, file_offsets

    def _build_split(
        self,
        stems: list[str],
        split_indices: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[
        dict[int, LazyDICHASUSTrajectoryDataset],
        dict[tuple[int, int], LazyDICHASUSSharedDataset],
        dict,
    ]:
        """Build datasets for a specific split (train/val/test).

        Creates both local (per-BS) and shared (inter-BS) datasets.

        Parameters
        ----------
        stems : list[str]
            File stems to use.
        split_indices : dict
            Dict with 'train' key containing indices for this split.
        rng : np.random.Generator
            Random generator for pair selection.

        Returns
        -------
        tuple
            (local_datasets, shared_datasets, idx_to_neg_pos)
        """
        # Build file record map (lazy loading index)
        file_record_map, timestamps_all, file_offsets = self._build_file_record_map(stems)

        # Build positive/negative index mapping
        idx_to_neg_pos = self._build_idx_to_neg_pos(
            timestamps_all, file_id=0, global_offset=0, rng=rng
        )

        # All indices are valid (no BS-specific filtering in DICHASUS)
        N_total = len(timestamps_all)
        valid_idxs = np.arange(N_total, dtype=np.int64)

        apb = self.antennas_per_bs
        local_datasets: dict[int, LazyDICHASUSTrajectoryDataset] = {}

        # Calibration path — only meaningful for single-file datasets
        offsets_path: str | None = str(self.data_dir / f'reftx-offsets-{stems[0]}.json')
        if len(stems) > 1:
            offsets_path = None

        # Create one dataset per virtual BS (keyed by virtual BS index)
        for virt_id, phys_ids in enumerate(self.virtual_bs_groups):
            local_datasets[virt_id] = LazyDICHASUSTrajectoryDataset(
                idx_to_neg_pos=idx_to_neg_pos,
                valid_idxs=split_indices.get('train', valid_idxs),
                file_record_map=file_record_map,
                timestamps=timestamps_all,
                file_offsets=file_offsets,
                antennas_per_bs=apb,
                n_bs=self.n_bs,
                bs_ids=phys_ids,
                pair_mode=self.cfg['pair_mode'],
                p_positive=float(self.cfg['p_positive']),
                preprocess=self.preprocess,
                offsets_path=offsets_path,
            )

        # Create shared datasets for virtual BS pairs (if configured)
        shared_datasets: dict[tuple[int, int], LazyDICHASUSSharedDataset] = {}
        all_anchor_idxs = np.unique([anchor for (_, anchor) in idx_to_neg_pos])

        # Filter to only anchors in this split
        train_anchor_idxs = all_anchor_idxs[
            np.isin(all_anchor_idxs, split_indices.get('train', all_anchor_idxs))
        ]

        for v1, v2 in self.edge_set:
            if len(train_anchor_idxs) == 0:
                continue

            shared_datasets[(v1, v2)] = LazyDICHASUSSharedDataset(
                shared_global_idxs=train_anchor_idxs,
                file_record_map=file_record_map,
                antennas_per_bs=apb,
                n_bs=self.n_bs,
                bs_ids_1=self.virtual_bs_groups[v1],
                bs_ids_2=self.virtual_bs_groups[v2],
                preprocess=self.preprocess,
                offsets_path=offsets_path,
            )

        return local_datasets, shared_datasets, idx_to_neg_pos

    def setup(self, stage: str | None = None) -> None:
        """Set up the data module for training/validation/testing.

        This method:
        1. Builds the file record map (lazy loading index)
        2. Creates train/val/test splits at sample level
        3. Creates LazyDICHASUSTrajectoryDataset for each BS
        4. Creates LazyDICHASUSSharedDataset for each BS pair in edge_set
        5. Computes feature_dim from a sample

        Parameters
        ----------
        stage : str | None
            'train', 'val', 'test', or None (setup all)
        """
        if getattr(self, '_setup_done', False):
            return

        # Get split ratios
        train_split = self.cfg.get('train_split')
        val_split = self.cfg.get('val_split')
        test_split = self.cfg.get('test_split')

        # Sample-level split (train/val/test from same files)
        if train_split is not None and val_split is not None and test_split is not None:
            # Build file record map (lazy - only builds index, not CSI data)
            file_record_map, timestamps_all, file_offsets = self._build_file_record_map(
                self.file_stems
            )

            # Create random permutation of indices
            N_total = len(timestamps_all)
            all_indices = np.arange(N_total, dtype=np.int64)

            # Use anchor_rng for splitting (same anchors across folds)
            self.anchor_rng.shuffle(all_indices)

            # Use triplet_rng for positive/negative sampling (different per fold)
            triplet_rng = np.random.default_rng(self.triplet_seed)

            # Compute split boundaries
            n_train = int(train_split * N_total)
            n_val = int(val_split * N_total)

            # Extract indices for each split
            train_indices = all_indices[:n_train]
            val_indices = all_indices[n_train : n_train + n_val]
            test_indices = all_indices[n_train + n_val :]

            # Build datasets for each split
            # Note: Each split gets its own dataset with filtered valid_idxs
            # But they share the same file_record_map (lazy loading works across splits)
            (
                self.train_local_dataset,
                self.train_shared_dataset,
                self.train_traj_dict,
            ) = self._build_split(self.file_stems, {'train': train_indices}, triplet_rng)

            (
                self.val_local_dataset,
                self.val_shared_dataset,
                self.val_traj_dict,
            ) = self._build_split(self.file_stems, {'train': val_indices}, triplet_rng)

            (
                self.test_local_dataset,
                self.test_shared_dataset,
                self.test_traj_dict,
            ) = self._build_split(self.file_stems, {'train': test_indices}, triplet_rng)

        # Compute feature dimension from a sample
        # This triggers the first lazy load (and caching)
        sample_csi = self.train_local_dataset[0]._H_from_global_index(
            list(self.train_local_dataset[0].idx_to_neg_pos.values())[0]['pos'][0]
        )
        self.feature_dim = csi_to_realvec(sample_csi, method=self.preprocess).shape[0]

        self._setup_done = True

    def _make_loaders(
        self,
        local: dict[int, LazyDICHASUSTrajectoryDataset],
        shared: dict[tuple[int, int], LazyDICHASUSSharedDataset],
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> CombinedLoader:
        """Create CombinedLoader from local and shared datasets.

        CombinedLoader handles multiple DataLoaders and iterates them
        in lockstep (min_size mode) or independently.

        Parameters
        ----------
        local : dict[int, LazyDICHASUSTrajectoryDataset]
            Per-BS datasets.
        shared : dict[tuple[int, int], LazyDICHASUSSharedDataset]
            Inter-BS datasets.
        shuffle : bool
            Whether to shuffle (typically False for train).
        drop_last : bool
            Whether to drop last incomplete batch.

        Returns
        -------
        CombinedLoader
            Combined loader containing all datasets.
        """
        loaders: dict = {}

        # Create DataLoader for each BS
        for bs_id, ds in local.items():
            loaders[bs_id] = DataLoader(
                ds,
                batch_size=self.cfg['batch_size'],
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.cfg['num_workers'],
                pin_memory=self.cfg['pin_memory'],
            )

        # Create DataLoader for each BS pair
        for (bs_1, bs_2), ds in shared.items():
            loaders[(bs_1, bs_2)] = DataLoader(
                ds,
                batch_size=self.cfg['batch_size'],
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.cfg['num_workers'],
                pin_memory=self.cfg['pin_memory'],
            )

        # max_size_cycle: iterate all loaders, cycle when any is exhausted
        return CombinedLoader(loaders, mode='max_size_cycle')

    def train_dataloader(self) -> CombinedLoader:
        """Create training dataloader.

        Returns
        -------
        CombinedLoader
            Combined loader with shuffle enabled, drop_last enabled.
        """
        return self._make_loaders(
            self.train_local_dataset,
            self.train_shared_dataset,
            shuffle=False,  # Shuffling handled by dataset's idx_to_neg_pos
            drop_last=True,  # Avoid incomplete batches in loss computation
        )

    def val_dataloader(self) -> CombinedLoader:
        """Create validation dataloader.

        Returns
        -------
        CombinedLoader
            Combined loader without shuffling.
        """
        return self._make_loaders(self.val_local_dataset, self.val_shared_dataset)

    def test_dataloader(self) -> CombinedLoader:
        """Create test dataloader.

        Returns
        -------
        CombinedLoader
            Combined loader without shuffling.
        """
        return self._make_loaders(self.test_local_dataset, self.test_shared_dataset)
