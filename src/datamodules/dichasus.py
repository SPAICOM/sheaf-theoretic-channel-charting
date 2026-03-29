"""
PyTorch Lightning DataModule for the DICHASUS real-world CSI dataset.

Unlike DeepMIMO (simulated + synthetic trajectories), this module:
1. Loads real TFRecord files from the DICHASUS distributed channel sounder.
2. Applies STO/CPO calibration via per-file offset JSON files.
3. Splits the full antenna array into per-BS sub-arrays.
4. Builds positive/negative sample pairs from timestamps (time-coherence window Tc),
   not from synthetic trajectories.
5. Treats each TFRecord file as one independent measurement campaign
   (no cross-file pairs).
6. Reuses TrajectoryCSIDataset and SharedTrajectoryCSIDataset unchanged,
   so the same orchestrator/agent pipeline works without modification.

DICHASUS antenna layout (cf0x datasets)
----------------------------------------
The raw CSI tensor has shape (total_antennas, 1024) with all BSs concatenated.
For the cf0x family the default split is 4 BSs × 8 antennas each (= 32 total).
This is configurable via ``n_bs`` and ``antennas_per_bs``.

Key differences from DeepMimoDataModule
----------------------------------------
* No trajectory generation: the UE movement is real and already recorded.
* Positive/negative pairs are defined by a time-coherence window
  (``Tc_pos`` / ``Tc_neg`` in seconds) rather than trajectory step windows.
* All BSs see the same positions simultaneously (no per-BS coverage filtering).
  ``valid_idxs`` is identical for every BS.
* File-level train/val/test split (each .tfrecords = one measurement campaign).
* Offset JSON files are served directly from the DICHASUS website.

Data structure per sample (DICHASUS cf0x)
------------------------------------------
    CSI        : (total_antennas, 1024) complex64
                 split into n_bs arrays of (antennas_per_bs, 1, 1024) to match (R, T, F)
    Position   : (2,) float64   XY from tachymeter ground truth
    Timestamp  : float32 scalar seconds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as l
import numpy as np
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .deepmimo import SharedTrajectoryCSIDataset, TrajectoryCSIDataset, _merge_defaults
from .utils import csi_to_realvec


class DICHASUSDataModule(l.LightningDataModule):
    """
    PyTorch Lightning DataModule for trajectory-based CSI learning on DICHASUS data.

    Mirrors the interface of ``DeepMimoDataModule`` so that the same
    orchestrators, agents, and training scripts work without modification.

    Parameters
    ----------
    dataset_cfg : DictConfig | dict | None
        Configuration overriding DEFAULTS.
    seed : int | None
        Global random seed.

    Attributes
    ----------
    n_agents : int
        Number of distributed BSs = ``n_bs``.
    feature_dim : int
        Dimension of the per-BS CSI feature vector (set during ``setup``).
    train/val/test_local_dataset : dict[int, TrajectoryCSIDataset]
        ``{bs_id: dataset}`` — one entry per BS, same format as DeepMimoDataModule.
    train/val/test_shared_dataset : dict[tuple[int,int], SharedTrajectoryCSIDataset]
        One entry per BS pair in ``edge_set``; shared positions = all anchor points
        (every BS sees every UE position simultaneously in DICHASUS).
    train/val/test_traj_dict : dict
        Positive/negative index mapping (idx_to_neg_pos) for each split.
    """

    # TFRecord DOIs (DaRUS) — offsets served directly from the DICHASUS website
    ALL_FILES: dict[str, str] = {
        'dichasus-cf02': 'doi:10.18419/darus-2854/8',
        'dichasus-cf03': 'doi:10.18419/darus-2854/9',
        'dichasus-cf04': 'doi:10.18419/darus-2854/10',
        'dichasus-cf05': 'doi:10.18419/darus-2854/11',
        'dichasus-cf06': 'doi:10.18419/darus-2854/12',
        'dichasus-cf07': 'doi:10.18419/darus-2854/13',
    }

    # Base URL for offset JSON calibration files (served by DICHASUS website)
    _OFFSETS_BASE_URL = 'https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/'

    DEFAULTS: dict[str, Any] = {
        'data_dir': 'dichasus',
        # List of file stems to use, e.g. ['dichasus-cf02', 'dichasus-cf03'].
        # None → use all available files.
        'files': None,
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        # Distributed BS antenna layout.
        # total_antennas = n_bs * antennas_per_bs must match the stored CSI shape.
        'n_bs': 4,
        'antennas_per_bs': 8,
        # Time-coherence windows (seconds).
        #   Positives  :  |Δt| ≤ Tc_pos
        #   Negatives  :  Tc_pos < |Δt| ≤ Tc_neg
        'Tc_pos': 1.5,
        'Tc_neg': 5.0,
        'pair_mode': 'triplet',  # 'triplet' or 'contrastive'
        'p_positive': 0.5,  # for contrastive mode
        'n_pos': None,  # max positives per anchor (None = all)
        'n_neg': None,  # max negatives per anchor (None = all)
        'preprocess': 'dichasus',
        # BS pairs for shared datasets (inter-BS alignment loss).
        # Format: list of [bs_id_1, bs_id_2] pairs.  [] = no shared datasets.
        'edge_set': [],
        # Sample-level train/val/test split (fractions).
        # If None, falls back to file-level split via train_files/val_files/test_files.
        # If set, all files are concatenated and split by triplets.
        'train_split': None,  # e.g., 0.7
        'val_split': None,  # e.g., 0.15
        'test_split': None,  # e.g., 0.15
        # File-level train/val/test split (deprecated if train_split is set).
        # Each is a list of file stems.
        'train_files': None,
        'val_files': None,
        'test_files': None,
        'train_seed': 27,
        'test_seed': 42,
        'val_seed': 123,
    }

    def __init__(
        self,
        dataset_cfg: DictConfig | dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)
        self.data_dir = Path(self.cfg['data_dir'])

        self.file_stems: list[str] = self.cfg['files'] or list(self.ALL_FILES.keys())

        self.n_bs: int = int(self.cfg['n_bs'])
        self.antennas_per_bs: int = int(self.cfg['antennas_per_bs'])
        self.total_antennas: int = self.n_bs * self.antennas_per_bs

        self.Tc_pos = float(self.cfg['Tc_pos'])
        self.Tc_neg = float(self.cfg['Tc_neg'])
        self.preprocess: str = self.cfg.get('preprocess', 'dichasus')
        self.n_agents: int = self.n_bs
        self.feature_dim: int | None = None

        self.edge_set: list[tuple[int, int]] = [tuple(e) for e in self.cfg.get('edge_set', [])]

        self._train_files: list[str] | None = self.cfg.get('train_files')
        self._val_files: list[str] | None = self.cfg.get('val_files')
        self._test_files: list[str] | None = self.cfg.get('test_files')

        self.rng = np.random.default_rng(seed)

        assert self.Tc_neg > self.Tc_pos, '"Tc_neg" must be greater than "Tc_pos"'

    # ------------------------------------------------------------------
    # Lightning: prepare_data
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download TFRecord files (DaRUS) and calibration offset JSONs (DICHASUS site)."""
        darus_base = (
            'https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId='
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

        from tqdm import tqdm

        for stem in self.file_stems:
            # TFRecord from DaRUS
            tfrecords_dest = self.data_dir / f'{stem}.tfrecords'
            if not tfrecords_dest.exists():
                self._download(darus_base + self.ALL_FILES[stem], tfrecords_dest, tqdm)

            # Offset JSON from DICHASUS website
            offsets_dest = self.data_dir / f'reftx-offsets-{stem}.json'
            if not offsets_dest.exists():
                self._download(
                    self._OFFSETS_BASE_URL + f'reftx-offsets-{stem}.json',
                    offsets_dest,
                    tqdm,
                )

    @staticmethod
    def _download(url: str, dest: Path, tqdm_cls) -> None:
        r = requests.get(url, stream=True, verify=False)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with (
            open(dest, 'wb') as f,
            tqdm_cls(total=total, unit='B', unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    # ------------------------------------------------------------------
    # TFRecord parsing + calibration
    # ------------------------------------------------------------------

    def _parse_tfrecord(
        self,
        stem: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and calibrate one DICHASUS TFRecord file.

        Parameters
        ----------
        stem : str
            File stem, e.g. ``'dichasus-cf02'``.

        Returns
        -------
        csi : np.ndarray, shape (N, total_antennas, 1, 1024), complex64
            Calibrated channel matrices.  The size-1 axis is the T (transmit)
            dimension expected by ``csi_to_realvec``.
        positions : np.ndarray, shape (N, 2), float64
            XY ground-truth positions from tachymeter.
        timestamps : np.ndarray, shape (N,), float32
            Sorted recording timestamps (seconds).
        """
        import tensorflow as tf  # deferred import — TF is not required at module load

        tfrecords_path = str(self.data_dir / f'{stem}.tfrecords')
        offsets_path = self.data_dir / f'reftx-offsets-{stem}.json'

        offsets = None
        if offsets_path.exists():
            with open(offsets_path) as f:
                offsets = json.load(f)

        total_ant = self.total_antennas
        subcarriers = 1024

        def _parse(proto):
            record = tf.io.parse_single_example(
                proto,
                {
                    'csi': tf.io.FixedLenFeature([], tf.string, default_value=''),
                    'pos-tachy': tf.io.FixedLenFeature([], tf.string, default_value=''),
                    'time': tf.io.FixedLenFeature([], tf.float32, default_value=0),
                },
            )
            csi = tf.ensure_shape(
                tf.io.parse_tensor(record['csi'], out_type=tf.float32),
                (total_ant, subcarriers, 2),
            )
            csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
            csi = tf.signal.fftshift(csi, axes=1)
            position = tf.ensure_shape(
                tf.io.parse_tensor(record['pos-tachy'], out_type=tf.float64), (3,)
            )
            time = tf.ensure_shape(record['time'], ())
            return csi, position[:2], time

        dataset = tf.data.TFRecordDataset(tfrecords_path).map(
            _parse, num_parallel_calls=tf.data.AUTOTUNE
        )

        # STO/CPO phase calibration (per antenna × per subcarrier)
        if offsets is not None:

            def _calibrate(csi, pos, time):
                sc_idx = tf.cast(tf.range(subcarriers), tf.float32)
                sto_offset = tf.tensordot(
                    tf.constant(offsets['sto'], dtype=tf.float32),
                    2.0 * np.pi * sc_idx / subcarriers,
                    axes=0,
                )
                cpo_offset = tf.tensordot(
                    tf.constant(offsets['cpo'], dtype=tf.float32),
                    tf.ones(subcarriers, dtype=tf.float32),
                    axes=0,
                )
                csi = csi * tf.exp(tf.complex(tf.zeros_like(sto_offset), sto_offset + cpo_offset))
                return csi, pos, time

            dataset = dataset.map(_calibrate, num_parallel_calls=tf.data.AUTOTUNE)

        csi_list, pos_list, time_list = [], [], []
        for csi, pos, time in dataset:
            csi_list.append(csi.numpy())  # (total_ant, 1024) complex64
            pos_list.append(pos.numpy())  # (2,) float64
            time_list.append(time.numpy())  # float32 scalar

        # Stack, then insert the T=1 transmit-antenna axis
        # Shape: (N, total_antennas, 1, 1024) — matches (N, R, T, F) convention
        csi_arr = np.stack(csi_list, axis=0)[:, :, np.newaxis, :]  # complex64
        positions = np.stack(pos_list, axis=0)  # float64
        timestamps = np.array(time_list, dtype=np.float32)

        # Sort by timestamp (measurements may have minor reordering)
        order = np.argsort(timestamps)
        return csi_arr[order], positions[order], timestamps[order]

    # ------------------------------------------------------------------
    # Positive/negative pair construction
    # ------------------------------------------------------------------

    def _build_idx_to_neg_pos(
        self,
        timestamps: np.ndarray,
        file_id: int,
        global_offset: int,
        rng: np.random.Generator,
    ) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        """Build the positive/negative lookup dict for one measurement file.

        Uses binary search on sorted timestamps → O(N log N).

        Positives  : j s.t. |t_j − t_i| ≤ Tc_pos  (j ≠ i)
        Negatives  : j s.t. Tc_pos < |t_j − t_i| ≤ Tc_neg

        ``file_id`` is used as the user_id key so entries from different files
        never collide, matching the (user_id, anchor_idx) key format of
        ``TrajectoryCSIDataset``.

        Parameters
        ----------
        timestamps : np.ndarray, shape (N,)
            Already sorted timestamps for this file.
        file_id : int
            Index of this file within the current split (used as user_id).
        global_offset : int
            Index of the first sample of this file in the concatenated array.
        rng : np.random.Generator
            For optional subsampling when n_pos / n_neg are set.

        Returns
        -------
        dict mapping (file_id, global_anchor_idx) → {'pos': ndarray, 'neg': ndarray}
        """
        N = len(timestamps)
        n_pos = self.cfg.get('n_pos')
        n_neg = self.cfg.get('n_neg')
        idx_to_neg_pos: dict = {}

        for i in range(N):
            t = timestamps[i]

            lo_pos = int(np.searchsorted(timestamps, t - self.Tc_pos, side='left'))
            hi_pos = int(np.searchsorted(timestamps, t + self.Tc_pos, side='right'))
            pos_local = np.arange(lo_pos, hi_pos, dtype=np.int64)
            pos_local = pos_local[pos_local != i]

            lo_neg = int(np.searchsorted(timestamps, t - self.Tc_neg, side='left'))
            hi_neg = int(np.searchsorted(timestamps, t + self.Tc_neg, side='right'))
            neg_candidates = np.arange(lo_neg, hi_neg, dtype=np.int64)
            neg_local = neg_candidates[(neg_candidates < lo_pos) | (neg_candidates >= hi_pos)]

            if len(pos_local) == 0 or len(neg_local) == 0:
                continue
            if n_pos is not None and len(pos_local) < n_pos:
                continue
            if n_neg is not None and len(neg_local) < n_neg:
                continue

            if n_pos is not None:
                pos_local = rng.choice(pos_local, size=n_pos, replace=False)
            if n_neg is not None:
                neg_local = rng.choice(neg_local, size=n_neg, replace=False)

            global_i = global_offset + i
            idx_to_neg_pos[(file_id, global_i)] = {
                'pos': pos_local + global_offset,
                'neg': neg_local + global_offset,
            }

        return idx_to_neg_pos

    # ------------------------------------------------------------------
    # Dataset construction (mirrors DeepMimoDataModule._shared_gen)
    # ------------------------------------------------------------------

    def _build_split(
        self,
        stems: list[str],
        rng: np.random.Generator,
    ) -> tuple[
        dict[int, TrajectoryCSIDataset],
        dict[tuple[int, int], SharedTrajectoryCSIDataset],
        dict,
    ]:
        """Load a list of TFRecord files and build local + shared datasets.

        Each BS ``k`` gets a ``TrajectoryCSIDataset`` whose channels are the
        antenna slice ``[:, k*apb:(k+1)*apb, :, :]``.

        Since all BSs observe the same UE positions simultaneously,
        ``valid_idxs`` is identical for all BSs and ``shared_traj_idxs``
        for a BS pair equals all anchor indices.

        Parameters
        ----------
        stems : list[str]
            File stems to load.
        rng : np.random.Generator
            For pair subsampling.

        Returns
        -------
        local_datasets : dict[int, TrajectoryCSIDataset]
        shared_datasets : dict[tuple[int,int], SharedTrajectoryCSIDataset]
        idx_to_neg_pos : dict
        """
        all_csi: list[np.ndarray] = []
        all_pos: list[np.ndarray] = []
        all_ts: list[np.ndarray] = []

        for stem in stems:
            csi, pos, ts = self._parse_tfrecord(stem)
            all_csi.append(csi)
            all_pos.append(pos)
            all_ts.append(ts)

        # Concatenated arrays: (N_total, total_antennas, 1, 1024)
        channels_all = np.concatenate(all_csi, axis=0)
        rx_pos = np.concatenate(all_pos, axis=0)  # (N_total, 2)

        # Build idx_to_neg_pos per file (no cross-file pairs)
        idx_to_neg_pos: dict = {}
        offset = 0
        for file_id, (csi_chunk, ts_chunk) in enumerate(zip(all_csi, all_ts)):
            file_map = self._build_idx_to_neg_pos(ts_chunk, file_id, offset, rng)
            idx_to_neg_pos.update(file_map)
            offset += len(ts_chunk)

        N_total = len(channels_all)
        valid_idxs = np.arange(N_total, dtype=np.int64)  # all BSs see all positions
        apb = self.antennas_per_bs

        # Per-BS local datasets: slice the antenna dimension
        local_datasets: dict[int, TrajectoryCSIDataset] = {}
        for bs_id in range(self.n_bs):
            channels_bs = channels_all[:, bs_id * apb : (bs_id + 1) * apb, :, :]
            local_datasets[bs_id] = TrajectoryCSIDataset(
                idx_to_neg_pos=idx_to_neg_pos,
                valid_idxs=valid_idxs,
                rx_pos=rx_pos,
                channels=channels_bs,
                pair_mode=self.cfg['pair_mode'],
                p_positive=float(self.cfg['p_positive']),
                preprocess=self.preprocess,
            )

        # Shared datasets: all anchor positions are shared (simultaneous capture)
        shared_datasets: dict[tuple[int, int], SharedTrajectoryCSIDataset] = {}
        all_anchor_idxs = np.unique([anchor for (_, anchor) in idx_to_neg_pos])

        for bs_1, bs_2 in self.edge_set:
            if len(all_anchor_idxs) == 0:
                continue

            shared_pos = rx_pos[all_anchor_idxs]
            channels_bs_1 = channels_all[all_anchor_idxs, bs_1 * apb : (bs_1 + 1) * apb, :, :]
            channels_bs_2 = channels_all[all_anchor_idxs, bs_2 * apb : (bs_2 + 1) * apb, :, :]

            shared_datasets[(bs_1, bs_2)] = SharedTrajectoryCSIDataset(
                idx_bs_1=bs_1,
                idx_bs_2=bs_2,
                shared_pos=shared_pos,
                channels_bs_1=channels_bs_1,
                channels_bs_2=channels_bs_2,
                shared_traj_idxs=all_anchor_idxs,
                preprocess=self.preprocess,
            )

        return local_datasets, shared_datasets, idx_to_neg_pos

    # ------------------------------------------------------------------
    # Lightning: setup
    # ------------------------------------------------------------------

    def _load_all_files(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Load all TFRecord files and concatenate them into one dataset.

        Returns
        -------
        channels_all : np.ndarray, shape (N_total, total_antennas, 1, 1024)
        rx_pos : np.ndarray, shape (N_total, 2)
        timestamps_all : np.ndarray, shape (N_total,)
        idx_to_neg_pos : dict
        """
        all_csi: list[np.ndarray] = []
        all_pos: list[np.ndarray] = []
        all_ts: list[np.ndarray] = []

        for stem in self.file_stems:
            csi, pos, ts = self._parse_tfrecord(stem)
            all_csi.append(csi)
            all_pos.append(pos)
            all_ts.append(ts)

        channels_all = np.concatenate(all_csi, axis=0)
        rx_pos = np.concatenate(all_pos, axis=0)
        timestamps_all = np.concatenate(all_ts, axis=0)

        rng = np.random.default_rng(self.cfg['train_seed'])
        idx_to_neg_pos = self._build_idx_to_neg_pos(
            timestamps_all, file_id=0, global_offset=0, rng=rng
        )

        return channels_all, rx_pos, timestamps_all, idx_to_neg_pos

    def _build_datasets_from_concatenated(
        self,
        channels_all: np.ndarray,
        rx_pos: np.ndarray,
        idx_to_neg_pos: dict,
        split_indices: dict[str, np.ndarray],
    ) -> tuple[
        dict[int, TrajectoryCSIDataset],
        dict[tuple[int, int], SharedTrajectoryCSIDataset],
        dict,
    ]:
        """Build local and shared datasets from concatenated data using split indices.

        Parameters
        ----------
        channels_all : np.ndarray
            Full concatenated CSI array.
        rx_pos : np.ndarray
            Full concatenated positions.
        idx_to_neg_pos : dict
            Full positive/negative index mapping.
        split_indices : dict
            Dict with 'train', 'val', 'test' keys, each containing np.ndarray of indices.

        Returns
        -------
        local_datasets : dict[int, TrajectoryCSIDataset]
        shared_datasets : dict[tuple[int,int], SharedTrajectoryCSIDataset]
        traj_dict : dict
        """
        apb = self.antennas_per_bs

        local_datasets: dict[int, TrajectoryCSIDataset] = {}
        for bs_id in range(self.n_bs):
            channels_bs = channels_all[:, bs_id * apb : (bs_id + 1) * apb, :, :]
            local_datasets[bs_id] = TrajectoryCSIDataset(
                idx_to_neg_pos=idx_to_neg_pos,
                valid_idxs=split_indices['train'],
                rx_pos=rx_pos,
                channels=channels_bs,
                pair_mode=self.cfg['pair_mode'],
                p_positive=float(self.cfg['p_positive']),
                preprocess=self.preprocess,
            )

        shared_datasets: dict[tuple[int, int], SharedTrajectoryCSIDataset] = {}
        all_anchor_idxs = np.unique([anchor for (_, anchor) in idx_to_neg_pos])
        train_anchor_idxs = all_anchor_idxs[np.isin(all_anchor_idxs, split_indices['train'])]

        for bs_1, bs_2 in self.edge_set:
            if len(train_anchor_idxs) == 0:
                continue

            shared_pos = rx_pos[train_anchor_idxs]
            channels_bs_1 = channels_all[train_anchor_idxs, bs_1 * apb : (bs_1 + 1) * apb, :, :]
            channels_bs_2 = channels_all[train_anchor_idxs, bs_2 * apb : (bs_2 + 1) * apb, :, :]

            shared_datasets[(bs_1, bs_2)] = SharedTrajectoryCSIDataset(
                idx_bs_1=bs_1,
                idx_bs_2=bs_2,
                shared_pos=shared_pos,
                channels_bs_1=channels_bs_1,
                channels_bs_2=channels_bs_2,
                shared_traj_idxs=train_anchor_idxs,
                preprocess=self.preprocess,
            )

        return local_datasets, shared_datasets, idx_to_neg_pos

    def setup(self, stage: str | None = None) -> None:
        """Load TFRecord files and construct train/val/test datasets.

        Populates:
        - ``train/val/test_local_dataset``   : ``{bs_id: TrajectoryCSIDataset}``
        - ``train/val/test_shared_dataset``  : ``{(bs1, bs2): SharedTrajectoryCSIDataset}``
        - ``train/val/test_traj_dict``       : idx_to_neg_pos dicts
        - ``feature_dim``                    : int
        """
        if getattr(self, '_setup_done', False):
            return

        train_split = self.cfg.get('train_split')
        val_split = self.cfg.get('val_split')
        test_split = self.cfg.get('test_split')

        if train_split is not None and val_split is not None and test_split is not None:
            channels_all, rx_pos, timestamps_all, idx_to_neg_pos = self._load_all_files()

            N_total = len(channels_all)
            all_indices = np.arange(N_total, dtype=np.int64)

            rng = np.random.default_rng(self.cfg['train_seed'])
            rng.shuffle(all_indices)

            n_train = int(train_split * N_total)
            n_val = int(val_split * N_total)

            train_indices = all_indices[:n_train]
            val_indices = all_indices[n_train : n_train + n_val]
            test_indices = all_indices[n_train + n_val :]

            split_indices = {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices,
            }

            (
                self.train_local_dataset,
                self.train_shared_dataset,
                self.train_traj_dict,
            ) = self._build_datasets_from_concatenated(
                channels_all, rx_pos, idx_to_neg_pos, {'train': train_indices}
            )

            (
                self.val_local_dataset,
                self.val_shared_dataset,
                self.val_traj_dict,
            ) = self._build_datasets_from_concatenated(
                channels_all, rx_pos, idx_to_neg_pos, {'train': val_indices}
            )

            (
                self.test_local_dataset,
                self.test_shared_dataset,
                self.test_traj_dict,
            ) = self._build_datasets_from_concatenated(
                channels_all, rx_pos, idx_to_neg_pos, {'train': test_indices}
            )

        else:
            train_stems = self._train_files
            val_stems = self._val_files
            test_stems = self._test_files

            if train_stems is None and val_stems is None and test_stems is None:
                n = len(self.file_stems)
                n_train = max(1, int(round(0.7 * n)))
                n_val = max(1, (n - n_train) // 2)
                train_stems = self.file_stems[:n_train]
                val_stems = self.file_stems[n_train : n_train + n_val]
                test_stems = self.file_stems[n_train + n_val :] or self.file_stems[-1:]

            if train_stems is None:
                raise ValueError('train_files must be specified for file-level split')
            if val_stems is None:
                raise ValueError('val_files must be specified for file-level split')
            if test_stems is None:
                raise ValueError('test_files must be specified for file-level split')

            (
                self.train_local_dataset,
                self.train_shared_dataset,
                self.train_traj_dict,
            ) = self._build_split(train_stems, np.random.default_rng(self.cfg['train_seed']))

            (
                self.val_local_dataset,
                self.val_shared_dataset,
                self.val_traj_dict,
            ) = self._build_split(val_stems, np.random.default_rng(self.cfg['val_seed']))

            (
                self.test_local_dataset,
                self.test_shared_dataset,
                self.test_traj_dict,
            ) = self._build_split(test_stems, np.random.default_rng(self.cfg['test_seed']))

        sample_csi = torch.from_numpy(self.train_local_dataset[0].channels[0])
        self.feature_dim = csi_to_realvec(sample_csi, method=self.preprocess).shape[0]

        self._setup_done = True

    # ------------------------------------------------------------------
    # Lightning: dataloaders
    # ------------------------------------------------------------------

    def _make_loaders(
        self,
        local: dict[int, TrajectoryCSIDataset],
        shared: dict[tuple[int, int], SharedTrajectoryCSIDataset],
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> CombinedLoader:
        loaders: dict = {}
        for bs_id, ds in local.items():
            loaders[bs_id] = DataLoader(
                ds,
                batch_size=self.cfg['batch_size'],
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.cfg['num_workers'],
                pin_memory=self.cfg['pin_memory'],
            )
        for (bs_1, bs_2), ds in shared.items():
            loaders[(bs_1, bs_2)] = DataLoader(
                ds,
                batch_size=self.cfg['batch_size'],
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=self.cfg['num_workers'],
                pin_memory=self.cfg['pin_memory'],
            )
        return CombinedLoader(loaders, mode='min_size')

    def train_dataloader(self) -> CombinedLoader:
        return self._make_loaders(
            self.train_local_dataset,
            self.train_shared_dataset,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self) -> CombinedLoader:
        return self._make_loaders(self.val_local_dataset, self.val_shared_dataset)

    def test_dataloader(self) -> CombinedLoader:
        return self._make_loaders(self.test_local_dataset, self.test_shared_dataset)
