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

from pathlib import Path
from typing import Any

import deepmimo as dm
import lightning as l
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from matplotlib.patches import Circle, Rectangle
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import DataLoader

from ..datasets import (
    SharedTrajectoryCSIDataset,
    TrajectoryCSIDataset,
    csi_to_realvec_ferrand,
)


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


class DeepMimoDataModule(l.LightningDataModule):
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
        'pin_memory': True,
        'compute_channels': {},
        'train_num_users': 1,
        'test_num_users': 1,
        'val_num_users': 1,
        'T_min': 20,
        'T_max': 60,
        'r_min': None,
        'r_max': None,
        'z_min': None,
        'z_max': None,
        'linear_len': (20.0, 120.0),
        'circle_diameter': (20.0, 120.0),
        'random_step': (1.0, 5.0),
        'random_keep_dir': 0.7,
        'heading_momentum': 0.8,  # 0=no memory, 1=fixed
        'trajectory_kind': 'full',
        'pair_mode': 'triplet',  # "triplet" or "contrastive"
        'in_window': 3,
        'out_window': 6,
        'bias_sampling': False,
        'coverage_area': 0.2,
        'include_same_user_outside_window': False,
        'p_positive': 0.5,  # for contrastive
        'n_pos': None,  # number of positives to sample per anchor (None = all)
        'n_neg': None,  # number of negatives to sample per anchor (None = all)
        'train_seed': 27,
        'test_seed': 42,
        'val_seed': 123,
        'edge_set': [],
        'exclude_buildings': False,  # filter out RX positions inside building footprints
        'streets_only': False,  # restrict feasible set to street areas only
        'streets_proximity_dist': 15.0,  # max distance (m) from a building to count as "street"
        'inscribed_ellipse': False,  # restrict to largest ellipse inside coverage
    }

    def __init__(
        self,
        dataset_cfg: DictConfig | dict[str, Any],
        seed: int | None = None,
    ):
        """
        Initialize the DeepMimoDataModule.

        Parameters
        ----------
        dataset_cfg : DictConfig | dict[str, Any]
            User configuration overriding DEFAULTS.
            Can be provided via Hydra/OmegaConf or as a normal dictionary.
        """
        super().__init__()

        # Merge user configuration with defaults
        self.cfg = _merge_defaults(self.DEFAULTS, dataset_cfg)

        # Casting
        self.in_window = int(self.cfg['in_window'])
        self.out_window = int(self.cfg['out_window'])
        self.bias_sampling = bool(self.cfg['bias_sampling'])
        self.z_min = self.cfg['z_min']
        self.z_max = self.cfg['z_max']
        self.r_min = self.cfg['r_min']
        self.r_max = self.cfg['r_max']
        self.T_min = self.cfg['T_min']
        self.T_max = self.cfg['T_max']
        self.coverage_area = self.cfg['coverage_area']
        self.trajectory_kind = self.cfg['trajectory_kind']
        self.rng = np.random.default_rng(seed)
        self.kinds = (
            'linear',
            'circular',
            'random',
            'full',
            'neighbor_linear',
            'neighbor_l',
        )

        self.edge_set = [tuple(edge) for edge in self.cfg['edge_set']]
        self.n_pos = self.cfg['n_pos']
        self.n_neg = self.cfg['n_neg']

        # Dataset placeholders (initialized during setup)
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.n_agents = 1
        self.feature_dim = None

        # Assertions
        assert self.out_window > self.in_window, (
            '"out_window" must always be grater than "in_window"'
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

    # ----------------- Building mask -----------------
    @staticmethod
    def _build_hull_equations(
        scene,
        min_hull_area: float = 0.0,
        max_hull_area: float = 1000.0,
    ) -> list[np.ndarray]:
        """
        Compute 2D convex hull equations for scene objects whose hull area
        falls within [min_hull_area, max_hull_area].

        In the Houston DeepMIMO scenario objects labelled 'buildings' include
        both actual buildings (≤ 342 m²) and road/highway surfaces (≥ 3420 m²).
        The default range [0, 1000] selects only real buildings.
        Passing min_hull_area=1000 / max_hull_area=inf selects roads only.

        Returns list of (M_i, 3) half-plane equation arrays [a,b,c]
        s.t. ax + by + c <= 0 defines the interior of the hull.
        """
        from scipy.spatial import ConvexHull

        hulls = []
        for b in scene.get_objects('buildings'):
            verts_xy = np.unique(np.array(b.vertices)[:, :2], axis=0)
            if len(verts_xy) < 3:
                continue
            try:
                hull = ConvexHull(verts_xy)
                if hull.volume < min_hull_area or hull.volume > max_hull_area:
                    continue
                hulls.append(hull.equations)
            except Exception:
                continue
        return hulls

    @staticmethod
    def _outside_buildings_mask(
        rx_pos_xy: np.ndarray, hull_equations: list[np.ndarray]
    ) -> np.ndarray:
        """
        Return a boolean mask of RX positions that lie outside all building footprints.

        Parameters
        ----------
        rx_pos_xy : np.ndarray
            Array of shape (N, 2) with XY coordinates of RX positions.
        hull_equations : list of np.ndarray
            Pre-computed 2D hull equations from _build_hull_equations().

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,), True where the position is NOT inside any building.
        """
        inside = np.zeros(len(rx_pos_xy), dtype=bool)
        for eqs in hull_equations:
            # eqs: (M, 3) — ax+by+c <= 0 for interior
            inside |= np.all(rx_pos_xy @ eqs[:, :2].T + eqs[:, 2] <= 0, axis=1)
        return ~inside

    @staticmethod
    def _compute_inscribed_ellipse(
        coverage_xy: np.ndarray,
        bs_positions_2d: np.ndarray,
        coverage_radii: np.ndarray,
        n_angle_samples: int = 720,
    ) -> tuple[np.ndarray, float, float]:
        """
        Find the largest axis-aligned ellipse inscribed in the union of BS
        coverage circles.

        The ellipse is centered at the centroid of ``coverage_xy`` and its
        aspect ratio is fixed to match the bounding box of those positions.
        A binary search on a common scale factor yields the maximum
        semi-axes (a along x, b along y) such that every point on the
        ellipse boundary lies within at least one coverage circle.

        Parameters
        ----------
        coverage_xy : np.ndarray, shape (N, 2)
            XY positions currently inside the union coverage area (used only
            to derive the centroid and bounding-box aspect ratio).
        bs_positions_2d : np.ndarray, shape (B, 2)
        coverage_radii : np.ndarray, shape (B,)
        n_angle_samples : int
            Number of boundary points sampled for containment checks.

        Returns
        -------
        center : np.ndarray, shape (2,)
        semi_a : float   semi-axis along x
        semi_b : float   semi-axis along y
        """
        center = coverage_xy.mean(axis=0)

        bbox_w = float(np.ptp(coverage_xy[:, 0]))
        bbox_h = float(np.ptp(coverage_xy[:, 1]))
        aspect = bbox_w / bbox_h  # a = aspect * b

        angles = np.linspace(0, 2 * np.pi, n_angle_samples, endpoint=False)
        # Unit-ellipse direction vectors: scale by (aspect, 1) so that when
        # multiplied by scalar s they give (a*cos θ, b*sin θ) = (aspect*s*cos θ, s*sin θ)
        dirs = np.stack([aspect * np.cos(angles), np.sin(angles)], axis=1)

        def all_in_union(scale: float) -> bool:
            pts = center + scale * dirs
            covered = np.zeros(n_angle_samples, dtype=bool)
            for bp, r in zip(bs_positions_2d, coverage_radii):
                covered |= np.linalg.norm(pts - bp, axis=1) <= r
            return bool(covered.all())

        # Safe upper bound: b can't exceed r_max (the ellipse can't be wider
        # than the largest coverage circle), so b_max = r_max and s = b.
        hi = float(coverage_radii.max())
        lo = 0.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if all_in_union(mid):
                lo = mid
            else:
                hi = mid

        return center, lo * aspect, lo

    # ----------------- Building crossing check -----------------
    def _segments_cross_buildings(self, p1: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask (K,) — True where the straight-line segment
        from p1 to candidates[k] crosses the interior of at least one
        building convex hull (Liang-Barsky algorithm).
        """
        K = len(candidates)
        blocked = np.zeros(K, dtype=bool)
        dirs = candidates - p1  # (K, 2)

        for eqs in self.building_hull_equations:
            # eqs: (M, 3)  —  n·x + c ≤ 0 defines the interior
            u = dirs @ eqs[:, :2].T  # (K, M)
            w = eqs[:, :2] @ p1 + eqs[:, 2]  # (M,)

            # u==0 & w>0  →  segment parallel to & outside this half-plane
            #   → cannot be inside hull  →  no intersection
            no_hit = np.any((np.abs(u) < 1e-10) & (w[None, :] > 1e-10), axis=1)

            safe_u = np.where(np.abs(u) > 1e-10, u, 1.0)
            t = -w[None, :] / safe_u  # (K, M)

            t_hi = np.where(u > 1e-10, t, 1.0).min(axis=1)
            t_lo = np.where(u < -1e-10, t, 0.0).max(axis=1)

            blocked |= ~no_hit & (t_lo <= t_hi + 1e-10)

        return blocked

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
        xy = np.atleast_2d(xy)
        _, idx_local = self.kdtree.query(xy, k=1)
        return idx_local.astype(np.int64)

    def _generate_full_coverage(self) -> np.ndarray:
        """
        Generate a trajectory that visits every feasible RX point exactly once.

        The trajectory is constructed via a randomized nearest-neighbor walk
        over the feasible RX positions, ensuring spatial continuity.

        Returns
        -------
        np.ndarray
            Array of RX indices of length len(self.valid_idxs_all)
            covering all feasible points exactly once.
        """

        N = len(self.rx_pos_all_masked)
        visited = np.zeros(N, dtype=bool)
        traj = np.empty(N, dtype=np.int64)
        current = int(self.rng.integers(0, N))

        for t in range(N):
            traj[t] = current
            visited[current] = True
            if t == N - 1:
                break
            candidates = [n for n in self.neighbor_graph[current] if not visited[n]]
            if not candidates:
                candidates = list(np.where(~visited)[0])
            current = int(self.rng.choice(candidates))

        return traj

    def _neighbor_step(
        self,
        current: int,
        heading: np.ndarray | None,
        bias: float = 4.0,
    ) -> tuple[int, np.ndarray | None]:
        """
        Take one step on the prebuilt neighbor graph, optionally biased toward heading.

        Parameters
        ----------
        current : int
            Local index (into rx_pos_all_masked) of the current position.
        heading : np.ndarray | None
            Unit vector (2,) for directional bias.
            If None, the next neighbor is chosen uniformly at random.
        bias : float
            Sharpness of the directional bias (higher → more collinear).

        Returns
        -------
        next_local : int
            Local index of the chosen next position.
        new_heading : np.ndarray | None
            Updated heading unit vector. Returns None if heading was None.
        """
        neigh = self.neighbor_graph[current]
        if len(neigh) == 0:
            return current, heading  # isolated node — stay put

        if heading is None:
            return int(self.rng.choice(neigh)), None

        p1 = self.rx_pos_all_masked[current]
        diffs = self.rx_pos_all_masked[neigh] - p1
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
        cosines = (diffs / norms) @ heading
        probs = np.exp(bias * cosines)
        probs /= probs.sum()
        chosen = int(self.rng.choice(neigh, p=probs))

        new_dir = self.rx_pos_all_masked[chosen] - p1
        norm = np.linalg.norm(new_dir)
        new_heading = new_dir / norm if norm > 1e-8 else heading
        return chosen, new_heading

    # ----------------- Trajectory generators -----------------
    def _rand_anchor_xy(self) -> np.ndarray:
        """
        Pick a random RX XY coordinate to serve as a trajectory anchor.

        Returns
        -------
        np.ndarray
            Selected XY coordinate (2,).
        """
        # TODO self.H_users is not available
        if self.bias_sampling:
            power = np.linalg.norm(self.H_users.reshape(len(self.H_users), -1), axis=1)
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
            One of:
            - 'linear'          : straight line, snapped to grid
            - 'circular'        : circular arc, snapped to grid
            - 'random'          : continuous random walk, snapped to grid
            - 'full'            : bounded random walk on k-NN neighbor graph
            - 'neighbor_linear' : directionally-biased walk on neighbor graph
            - 'neighbor_l'      : L-shaped walk (two segments, ~90° turn)
        T : int
            Desired trajectory length (number of RX index steps).

        Returns
        -------
        np.ndarray
            Array of global RX indices representing the trajectory.
        """
        match kind:
            case 'linear':
                start = self._rand_anchor_xy()
                L = float(self.rng.uniform(*self.cfg['linear_len']))
                ang = float(self.rng.uniform(0, 2 * np.pi))
                end = start + L * np.array([np.cos(ang), np.sin(ang)])
                s = np.linspace(0.0, 1.0, T)
                xy = start * (1 - s)[:, None] + end * s[:, None]
                return self._snap(xy)

            case 'circular':
                # Oriented circular random walk on the k-NN graph.
                # The heading at each step is the tangent to the current
                # position w.r.t. a fixed center, so the path orbits the
                # center while the radius drifts naturally via the random
                # walk.  Rotation direction (CW / CCW) is chosen randomly.
                N = len(self.rx_pos_all_masked)
                current = int(self.rng.integers(0, N))
                start_pos = self.rx_pos_all_masked[current]

                # Center placed at a random direction / distance from start
                r_init = float(self.rng.uniform(*self.cfg['circle_diameter'])) / 2.0
                ang_to_center = float(self.rng.uniform(0, 2 * np.pi))
                center = start_pos + r_init * np.array(
                    [np.cos(ang_to_center), np.sin(ang_to_center)]
                )

                # CW (-1) or CCW (+1)
                direction = float(self.rng.choice([-1.0, 1.0]))

                traj = np.empty(T, dtype=np.int64)
                stuck_buf = np.empty((self._stuck_interval, 2))
                for t in range(T):
                    traj[t] = current
                    stuck_buf[t % self._stuck_interval] = self.rx_pos_all_masked[current]
                    if (
                        t > 0
                        and t % self._stuck_interval == 0
                        and stuck_buf.std(axis=0).max() < self._stuck_threshold
                    ):
                        # Stuck: reset orbit center near current position with new radius/direction
                        start_pos = self.rx_pos_all_masked[current]
                        r_init = float(self.rng.uniform(*self.cfg['circle_diameter'])) / 2.0
                        ang_to_center = float(self.rng.uniform(0, 2 * np.pi))
                        center = start_pos + r_init * np.array(
                            [np.cos(ang_to_center), np.sin(ang_to_center)]
                        )
                        direction = float(self.rng.choice([-1.0, 1.0]))
                    radius_vec = self.rx_pos_all_masked[current] - center
                    norm = np.linalg.norm(radius_vec)
                    if norm < 1e-8:
                        ang = float(self.rng.uniform(0, 2 * np.pi))
                        heading = np.array([np.cos(ang), np.sin(ang)])
                    else:
                        r_hat = radius_vec / norm
                        # Tangent: r_hat rotated 90° in chosen direction
                        heading = direction * np.array([-r_hat[1], r_hat[0]])
                    current, _ = self._neighbor_step(current, heading)
                return traj

            case 'random':
                xy = np.empty((T, 2), dtype=np.float64)
                xy[0] = self._rand_anchor_xy()
                prev_dir = None
                for t in range(1, T):
                    step = float(self.rng.uniform(*self.cfg['random_step']))
                    if prev_dir is None or self.rng.random() > self.cfg['random_keep_dir']:
                        ang = float(self.rng.uniform(0, 2 * np.pi))
                    else:
                        ang = float(np.arctan2(prev_dir[1], prev_dir[0]) + self.rng.normal(0, 0.5))
                    d = np.array([np.cos(ang), np.sin(ang)])
                    xy[t] = xy[t - 1] + step * d
                    prev_dir = d
                return self._snap(xy)

            case 'full':
                # Bounded random walk on the k-NN neighbor graph.
                # Visits exactly T feasible points w/o coverage requirement.
                N = len(self.rx_pos_all_masked)
                current = int(self.rng.integers(0, N))
                traj = np.empty(T, dtype=np.int64)
                for t in range(T):
                    traj[t] = current
                    current, _ = self._neighbor_step(current, heading=None)
                return traj

            case 'neighbor_linear':
                # Directionally-biased walk: prefer neighbors aligned with a
                # persistent heading → quasi-straight trajectory on the graph.
                N = len(self.rx_pos_all_masked)
                current = int(self.rng.integers(0, N))
                ang = float(self.rng.uniform(0, 2 * np.pi))
                heading = np.array([np.cos(ang), np.sin(ang)])
                traj = np.empty(T, dtype=np.int64)
                stuck_buf = np.empty((self._stuck_interval, 2))
                for t in range(T):
                    traj[t] = current
                    stuck_buf[t % self._stuck_interval] = self.rx_pos_all_masked[current]
                    if (
                        t > 0
                        and t % self._stuck_interval == 0
                        and stuck_buf.std(axis=0).max() < self._stuck_threshold
                    ):
                        # Stuck: pick a fresh random heading
                        ang = float(self.rng.uniform(0, 2 * np.pi))
                        heading = np.array([np.cos(ang), np.sin(ang)])
                    current, heading = self._neighbor_step(current, heading)
                return traj

            case 'neighbor_l':
                # L-shaped trajectory: two quasi-linear segments with a ~90°
                # turn between them. The turn direction (left/right) is random.
                N = len(self.rx_pos_all_masked)
                current = int(self.rng.integers(0, N))
                T1, T2 = T // 2, T - T // 2
                ang1 = float(self.rng.uniform(0, 2 * np.pi))
                sign = float(self.rng.choice([-1.0, 1.0]))
                turn = sign * (np.pi / 2 + float(self.rng.normal(0, 0.26)))
                headings = [
                    np.array([np.cos(ang1), np.sin(ang1)]),
                    np.array([np.cos(ang1 + turn), np.sin(ang1 + turn)]),
                ]
                traj = np.empty(T, dtype=np.int64)
                stuck_buf = np.empty((self._stuck_interval, 2))
                offset = 0
                for length, heading in zip([T1, T2], headings):
                    for step in range(length):
                        t_global = offset + step
                        traj[t_global] = current
                        stuck_buf[t_global % self._stuck_interval] = self.rx_pos_all_masked[current]
                        if (
                            t_global > 0
                            and t_global % self._stuck_interval == 0
                            and stuck_buf.std(axis=0).max() < self._stuck_threshold
                        ):
                            ang = float(self.rng.uniform(0, 2 * np.pi))
                            heading = np.array([np.cos(ang), np.sin(ang)])
                        current, heading = self._neighbor_step(current, heading)
                    offset += length
                return traj

            case _:
                raise RuntimeError(
                    f'Trajectory kind "{kind}" is not supported. '
                    'Choose from: "linear", "circular", "random", '
                    '"full", "neighbor_linear", "neighbor_l".'
                )

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

    def _shared_gen(self, num_users) -> tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
        """Generate shared dataset for all users.

        Creates triplet datasets (anchor, positive, negative samples) for
        each user based on their trajectory data.

        Parameters
        ----------
        num_users : int
            Number of users to generate data for.

        Returns
        -------
        tuple[dict[Any, Any], dict[Any, Any], dict[Any, Any]]
            Tuple containing:
            - Dictionary mapping user IDs to anchor data
            - Dictionary mapping user IDs to positive data
            - Dictionary mapping user IDs to negative data
        """
        idx_to_neg_pos = {}
        # user_id → full ordered trajectory (for plotting)
        traj_sequences: dict[int, np.ndarray] = {}
        for user_id in range(num_users):
            # Random trajectory length
            T_requested = int(self.rng.integers(self.T_min, self.T_max + 1))

            # Randomly pick trajectory kind
            kind = (
                self.kinds[int(self.rng.integers(0, len(self.kinds)))]
                if self.trajectory_kind is None
                else self.trajectory_kind
            )

            # Generate trajectory of RX indices
            rx_idxs = self._generate_one(kind, T_requested)
            traj_sequences[user_id] = rx_idxs
            T_actual = len(rx_idxs)

            for t in range(T_actual):
                # Skip trajectory endpoints (within out_window of start/end)
                if t < self.out_window or t >= T_actual - self.out_window:
                    continue

                anchor_rx = rx_idxs[t]

                # Positives: trajectory-time-adjacent steps within in_window
                pos_left = rx_idxs[max(0, t - self.in_window) : t]
                pos_right = rx_idxs[t + 1 : t + self.in_window + 1]
                pos = np.concatenate([pos_left, pos_right])

                # Negatives: steps outside in_window but within out_window
                neg_left = rx_idxs[max(0, t - self.out_window) : max(0, t - self.in_window)]
                neg_right = rx_idxs[t + self.in_window + 1 : t + self.out_window + 1]
                neg = np.concatenate([neg_left, neg_right])

                if len(pos) == 0 or len(neg) == 0:
                    continue
                if self.n_pos is not None and len(pos) < self.n_pos:
                    continue
                if self.n_neg is not None and len(neg) < self.n_neg:
                    continue

                if self.n_pos is not None:
                    pos = pos[self.rng.choice(len(pos), size=self.n_pos, replace=False)]
                if self.n_neg is not None:
                    neg = neg[self.rng.choice(len(neg), size=self.n_neg, replace=False)]

                idx_to_neg_pos[(user_id, anchor_rx)] = {
                    'pos': pos,
                    'neg': neg,
                }

        local_datasets = {}
        shared_datasets = {}

        # idx_to_neg_pos is {
        #   (user_id, anchor_idx):
        #       {'pos': pos_idxs,
        #       'neg': neg_idxs}
        # }
        # (anchor_idx, pos_idxs, neg_idxs) are all in coords w.r.t. rx_idxs
        # rx_idxs represents the indexing of rx_pos_all_masked (positions of the union area)
        # valid_idxs is WRONG
        # In local datasets I'm indexing the positions and channels based on the union of
        # the coverage areas rx_pos_all_masked
        for base_station, _ in enumerate(self.ds):
            local_datasets[base_station] = TrajectoryCSIDataset(
                idx_to_neg_pos=idx_to_neg_pos,
                valid_idxs=self.bs_coords[base_station]['valid_idxs'],
                # rx_pos=self.bs_coords[base_station]['rx_pos'],
                rx_pos=self.rx_pos_all_masked,
                channels=self.bs_coords[base_station]['channels'],
            )

        # Shared datasets: only trajectory anchor points visible to both BSes.
        # anchor_rx values in idx_to_neg_pos are indices into rx_pos_all_masked,
        # matching the indexing of bs_coords[bs_id]['valid_idxs'] and ['channels'].
        all_traj_idxs = np.unique([anchor_rx for (_, anchor_rx) in idx_to_neg_pos])

        for bs_1, bs_2 in self.edge_set:
            bs1_valid = set(self.bs_coords[bs_1]['valid_idxs'].tolist())
            bs2_valid = set(self.bs_coords[bs_2]['valid_idxs'].tolist())

            shared_traj_idxs = np.array(
                [idx for idx in all_traj_idxs if idx in bs1_valid and idx in bs2_valid],
                dtype=np.int64,
            )

            if len(shared_traj_idxs) == 0:
                continue

            shared_pos = self.rx_pos_all_masked_3d[shared_traj_idxs]
            channels_bs_1 = self.bs_coords[bs_1]['channels'][shared_traj_idxs]
            channels_bs_2 = self.bs_coords[bs_2]['channels'][shared_traj_idxs]

            shared_datasets[(bs_1, bs_2)] = SharedTrajectoryCSIDataset(
                idx_bs_1=bs_1,
                idx_bs_2=bs_2,
                shared_pos=shared_pos,
                channels_bs_1=channels_bs_1,
                channels_bs_2=channels_bs_2,
                shared_traj_idxs=shared_traj_idxs,
            )

        return (
            local_datasets,
            shared_datasets,
            idx_to_neg_pos,
            traj_sequences,
        )

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

        if getattr(self, '_setup_done', False):
            return

        # Load DeepMIMO scenario
        self.ds = dm.load(self.cfg['scenario'])

        # Channel computation arguments
        ch_kwargs = self.cfg.get('compute_channels') or {}

        max_subcarriers = self.cfg.get('compute_channels').get('max_subcarriers', 1)

        del ch_kwargs['max_subcarriers']

        ch_kwargs['ue_antenna']['shape'] = np.array(
            ch_kwargs.get('ue_antenna', {}).get('shape', [1, 1])
        )
        ch_kwargs['ofdm'] = {'selected_subcarriers': np.arange(max_subcarriers)}

        self.n_agents = len(self.ds.bs_pos)

        self.rx_pos_all = self.ds[0].rx_pos if len(self.ds.rx_pos) == 3 else self.ds.rx_pos
        self.valid_rx_pos = {}
        union_mask = np.zeros(self.rx_pos_all.shape[0], dtype=bool)

        # TODO: adjust for a single BS
        self.bs_coords = {}
        for bs_id, base_station in enumerate(self.ds):
            mask = np.ones(len(self.rx_pos_all), dtype=bool)
            base_station.compute_channels(**ch_kwargs)

            # ------- Filter coverage area basestation -------
            bs_pos = base_station.bs_pos

            d = np.linalg.norm(self.rx_pos_all - bs_pos, axis=1)

            if self.r_min is not None:
                mask &= d >= float(self.r_min)

            assert self.coverage_area >= 0 and self.coverage_area <= 1, (
                '"coverage_area" must be between 0 and 1 for a BS.'
            )
            if self.r_max is None:
                xmin, ymin = self.rx_pos_all[:, :2].min(axis=0)
                xmax, ymax = self.rx_pos_all[:, :2].max(axis=0)
                r_max = self.coverage_area * min(xmax - xmin, ymax - ymin)

            mask &= d <= float(r_max)

            # ------- Filter in the hight -------
            if self.z_min is not None:
                mask &= self.rx_pos_all[:, 2] >= float(self.z_min)
            if self.z_max is not None:
                mask &= self.rx_pos_all[:, 2] <= float(self.z_max)

            self.bs_coords[bs_id] = {
                'rx_pos': self.rx_pos_all[np.where(mask)[0]],
                'mask': mask,
                'coverage_radius': float(r_max),
                'coverage_points': int(np.sum(mask)),
                'bs_pos_2d': np.array(bs_pos).squeeze()[:2],
            }
            union_mask |= mask

        # ------- Optionally exclude positions inside building footprints -------
        # ------- Optionally restrict to inscribed ellipse inside coverage union -------
        self.ellipse_params: tuple[np.ndarray, float, float] | None = None
        if self.cfg.get('inscribed_ellipse', False):
            coverage_xy = self.rx_pos_all[union_mask][:, :2]
            bs_pos_2d = np.stack([self.bs_coords[i]['bs_pos_2d'] for i in range(self.n_agents)])
            radii = np.array([self.bs_coords[i]['coverage_radius'] for i in range(self.n_agents)])
            center, semi_a, semi_b = self._compute_inscribed_ellipse(coverage_xy, bs_pos_2d, radii)
            self.ellipse_params = (center, semi_a, semi_b)
            ellipse_mask = (
                (self.rx_pos_all[:, 0] - center[0]) ** 2 / semi_a**2
                + (self.rx_pos_all[:, 1] - center[1]) ** 2 / semi_b**2
            ) <= 1.0
            union_mask &= ellipse_mask

        self.building_hull_equations: list[np.ndarray] = []
        _scene = getattr(self.ds, 'scene', None) or getattr(self.ds[0], 'scene', None)
        if self.cfg.get('exclude_buildings', False) and _scene is not None:
            self.building_hull_equations = self._build_hull_equations(_scene)
            outside = self._outside_buildings_mask(
                self.rx_pos_all[:, :2], self.building_hull_equations
            )
            union_mask &= outside

        if self.cfg.get('streets_only', False):
            # Streets = outdoor positions within `streets_proximity_dist` metres of a
            # building.  Streets are flanked by buildings; parks/plazas are not.
            # streets_only always implies exclude_buildings (buildings must be excluded
            # from the feasible set and the crossing check must be active).
            streets_dist = float(self.cfg.get('streets_proximity_dist', 15.0))
            if not self.building_hull_equations and _scene is not None:
                self.building_hull_equations = self._build_hull_equations(_scene)
                outside = self._outside_buildings_mask(
                    self.rx_pos_all[:, :2], self.building_hull_equations
                )
                union_mask &= outside
            _bldg_hulls = self.building_hull_equations
            if _bldg_hulls:
                _inside_bldg = ~self._outside_buildings_mask(self.rx_pos_all[:, :2], _bldg_hulls)
                _bldg_xy = self.rx_pos_all[_inside_bldg, :2]
                if len(_bldg_xy) > 0:
                    from scipy.spatial import cKDTree as _cKDT

                    _bldg_tree = _cKDT(_bldg_xy)
                    _dists, _ = _bldg_tree.query(self.rx_pos_all[:, :2], k=1)
                    union_mask &= _dists < streets_dist

        self.valid_idxs_all = np.where(union_mask)[0]
        self.rx_pos_all_masked_3d = self.rx_pos_all[self.valid_idxs_all]
        self.rx_pos_all_masked = self.rx_pos_all_masked_3d[:, :2]
        self.kdtree = KDTree(self.rx_pos_all_masked)
        self.feasible_centroid = self.rx_pos_all_masked.mean(axis=0)
        # Typical inter-point distance (median 1st-neighbor distance), used for stuck detection
        _sample = self.rx_pos_all_masked[: min(500, len(self.rx_pos_all_masked))]
        self.typical_step_dist = float(np.median(self.kdtree.query(_sample, k=2)[0][:, 1]))
        # Stuck detector: check every N steps; restart heading if spatial std < threshold
        self._stuck_interval = int(self.cfg.get('stuck_check_interval', 200))
        self._stuck_threshold = self.typical_step_dist * float(
            self.cfg.get('stuck_threshold_steps', 8)
        )

        # Build 1-hop neighbor graph once: edges connect points within max_step_dist
        # that are not separated by a building.  Used by all trajectory generators.
        _max_step_dist = self.typical_step_dist * float(self.cfg.get('max_step_multiplier', 1.5))
        _N = len(self.rx_pos_all_masked)
        from collections import defaultdict as _dd

        _adj: dict[int, list[int]] = _dd(list)
        for _i, _j in self.kdtree.query_pairs(_max_step_dist):
            _adj[_i].append(_j)
            _adj[_j].append(_i)
        self.neighbor_graph: list[np.ndarray] = []
        for _i in range(_N):
            _nbrs = np.array(_adj[_i], dtype=np.int64)
            if len(_nbrs) > 0 and self.building_hull_equations:
                _blocked = self._segments_cross_buildings(
                    self.rx_pos_all_masked[_i], self.rx_pos_all_masked[_nbrs]
                )
                _nbrs = _nbrs[~_blocked]
            self.neighbor_graph.append(_nbrs)

        # Valid idxs for each BS in the self.rx_pos_all
        for bs_id, base_station in enumerate(self.ds):
            mask = np.ones(len(self.rx_pos_all_masked), dtype=bool)
            # ------- Filter coverage area basestation -------
            bs_pos = base_station.bs_pos

            d = np.linalg.norm(self.rx_pos_all_masked_3d - bs_pos, axis=1)
            mask &= d <= self.bs_coords[bs_id]['coverage_radius']
            local_valid_idxs = np.where(mask)[0]
            self.bs_coords[bs_id]['channels'] = base_station.channels[self.valid_idxs_all]
            self.bs_coords[bs_id]['valid_idxs'] = local_valid_idxs
            self.bs_coords[bs_id]['coverage_points'] = len(local_valid_idxs)

        # ---------------------------------------------------------------
        #                Compute dataset split sizes
        # ---------------------------------------------------------------

        train_num_users = int(self.cfg['train_num_users'])
        test_num_users = int(self.cfg['test_num_users'])
        val_num_users = int(self.cfg['val_num_users'])

        # ---------------------------------------------------------------
        #                   Create the Datasets
        # ---------------------------------------------------------------

        # Training dataset
        self.rng = np.random.default_rng(self.cfg['train_seed'])
        (
            self.train_local_dataset,
            self.train_shared_dataset,
            self.train_traj_dict,
            self.train_traj_seqs,
        ) = self._shared_gen(train_num_users)

        # Test dataset
        self.rng = np.random.default_rng(self.cfg['test_seed'])
        (
            self.test_local_dataset,
            self.test_shared_dataset,
            self.test_traj_dict,
            self.test_traj_seqs,
        ) = self._shared_gen(test_num_users)

        # Validation dataset
        self.rng = np.random.default_rng(self.cfg['val_seed'])
        (
            self.val_local_dataset,
            self.val_shared_dataset,
            self.val_traj_dict,
            self.val_traj_seqs,
        ) = self._shared_gen(val_num_users)

        sample_ch = torch.from_numpy(self.ds[0].channels[self.valid_idxs_all[0]])
        self.feature_dim = csi_to_realvec_ferrand(sample_ch).shape[0]

        self._setup_done = True
        return None

    def train_dataloader(self) -> CombinedLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader used during model training.
        """

        loaders = {}
        for base_station in self.train_local_dataset:
            loaders[base_station] = DataLoader(
                self.train_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
                drop_last=True,
            )

        for bs_1, bs_2 in self.train_shared_dataset:
            loaders[(bs_1, bs_2)] = DataLoader(
                self.train_shared_dataset[(bs_1, bs_2)],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
                drop_last=True,
            )

        return CombinedLoader(loaders, mode='max_size_cycle')

    def test_dataloader(self) -> CombinedLoader:
        """
        Create the test DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during testing/evaluation.
        """
        loaders = {}
        for base_station in self.test_local_dataset:
            loaders[base_station] = DataLoader(
                self.test_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
            )

        for bs_1, bs_2 in self.test_shared_dataset:
            loaders[(bs_1, bs_2)] = DataLoader(
                self.test_shared_dataset[(bs_1, bs_2)],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
            )

        return CombinedLoader(loaders, mode='max_size_cycle')

    def val_dataloader(self) -> CombinedLoader:
        """
        Create the validation DataLoader.

        Returns
        -------
        DataLoader
            DataLoader used during validation.
        """
        loaders = {}
        for base_station in self.val_local_dataset:
            loaders[base_station] = DataLoader(
                self.val_local_dataset[base_station],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
            )

        for bs_1, bs_2 in self.val_shared_dataset:
            loaders[(bs_1, bs_2)] = DataLoader(
                self.val_shared_dataset[(bs_1, bs_2)],
                batch_size=self.cfg['batch_size'],
                shuffle=False,
            )

        return CombinedLoader(loaders, mode='max_size_cycle')

    def plot_bs_coverage(
        self,
        plot_name: str | None = None,
    ) -> None:
        """
        Plot the coverage area of each base station together with RX positions.
        Useful for debugging coverage filtering.
        """

        rx_xy = self.rx_pos_all[:, :2]

        xmin, ymin = rx_xy.min(axis=0)
        xmax, ymax = rx_xy.max(axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))

        # RX grid bounding box (kept white)
        grid_rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
        )
        ax.add_patch(grid_rect)

        # color cycle
        colors = plt.cm.tab10.colors

        for bs_id, base_station in enumerate(self.ds):
            color = colors[bs_id % len(colors)]

            bs_pos = base_station.bs_pos.squeeze()
            center = bs_pos[:2]

            radius = self.bs_coords[bs_id]['coverage_radius']

            circle = Circle(
                center,
                radius,
                edgecolor=color,
                facecolor=color,
                alpha=0.18,
                linewidth=2,
            )

            # clip circle to RX grid
            circle.set_clip_path(grid_rect)

            ax.add_patch(circle)

            # BS marker
            ax.scatter(
                center[0],
                center[1],
                marker='^',
                s=160,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                label=f'BS {bs_id}',
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Base Station Coverage Areas')

        ax.legend()
        plt.tight_layout()
        plot_name = f'{self.cfg["scenario"]}_bs_coverage.png' if plot_name is None else plot_name
        plt.savefig(plot_name, dpi=300)
        plt.close()

        print(f'Coverage plot saved to: {plot_name}')

    def plot_trajectories(
        self,
        n: int,
        stage: str = 'train',
        plot_name: str | None = None,
        show_map: bool = True,
        n_clusters: int | None = None,
        sequential_shading: bool = False,
    ) -> None:
        """
        Plot the first n trajectories from a stage dataset over BS coverage.

        Parameters
        ----------
        n : int
            Number of trajectories to plot.
        stage : str
            One of 'train', 'val', 'test'.
        plot_name : str | None
            Output file name.
            Defaults to '<scenario>_<stage>_trajectories.png'.
        show_map : bool
            If True (default), render the DeepMIMO scenario map (buildings,
            terrain, etc.) as a 2D background layer.  Falls back gracefully
            when the scene is unavailable.
        n_clusters : int | None
            If provided, cluster all trajectory points with k-means using this
            many clusters and color each point by its cluster label instead of
            by trajectory.  Default is None (color by trajectory).
        sequential_shading : bool
            If True, color each trajectory's points from light (start of path)
            to dark (end of path), showing travel direction.  Ignored when
            n_clusters is set.  Default is False.
        """
        assert stage in ('train', 'val', 'test'), "stage must be one of 'train', 'val', 'test'"

        # Use the full ordered trajectory sequences (not the deduplicated traj_dict,
        # which loses revisited positions and causes apparent jumps on long trajectories)
        traj_seqs = {
            'train': self.train_traj_seqs,
            'val': self.val_traj_seqs,
            'test': self.test_traj_seqs,
        }[stage]

        selected_users = sorted(traj_seqs.keys())[:n]
        actual_n = len(selected_users)
        # user_points[uid] = full ordered rx_idx sequence for that user
        user_points: dict[int, np.ndarray] = {uid: traj_seqs[uid] for uid in selected_users}

        # --- Figure setup ---
        rx_xy = self.rx_pos_all[:, :2]
        xmin, ymin = rx_xy.min(axis=0)
        xmax, ymax = rx_xy.max(axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Render scenario map as background layer
        has_map = False
        if show_map:
            scene = getattr(self.ds, 'scene', None) or getattr(self.ds[0], 'scene', None)
            if scene is not None:
                try:
                    n_patches_before = len(ax.patches)
                    scene.plot(proj_3D=False, ax=ax, title=False, legend=False)
                    has_map = True
                    # Convert every scene patch to grayscale with low alpha so
                    # it reads as an unobtrusive background layer.
                    for patch in ax.patches[n_patches_before:]:
                        r, g, b, _ = patch.get_facecolor()
                        gray = 0.299 * r + 0.587 * g + 0.114 * b
                        patch.set_facecolor((gray, gray, gray, 0.05))
                        patch.set_edgecolor('none')
                except Exception:
                    pass  # scene unavailable for this scenario — skip silently

        # RX grid bounding box — transparent face when map is shown so the
        # scene image stays visible; white otherwise.
        grid_rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor='none' if has_map else 'white',
            edgecolor='black',
            linewidth=1.5,
        )
        ax.add_patch(grid_rect)

        # Draw BS coverage circles
        bs_colors = plt.cm.tab10.colors
        legend_handles = []
        for bs_id, base_station in enumerate(self.ds):
            color = bs_colors[bs_id % len(bs_colors)]
            bs_pos = base_station.bs_pos.squeeze()
            center = bs_pos[:2]
            radius = self.bs_coords[bs_id]['coverage_radius']

            circle = Circle(
                center,
                radius,
                edgecolor=color,
                facecolor=color,
                alpha=0.45,
                linewidth=2,
            )
            circle.set_clip_path(grid_rect)
            ax.add_patch(circle)

            handle = ax.scatter(
                center[0],
                center[1],
                marker='^',
                s=160,
                color=color,
                edgecolor='black',
                linewidth=0.5,
                label=f'BS {bs_id}',
                zorder=5,
            )
            legend_handles.append(handle)

        # --- Optional k-means cluster coloring ---
        cluster_labels: np.ndarray | None = None
        if n_clusters is not None:
            from sklearn.cluster import KMeans

            all_rx_idxs = [rx for uid in selected_users for rx in user_points[uid]]
            all_positions = self.rx_pos_all_masked[all_rx_idxs]
            km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            flat_labels = km.fit_predict(all_positions)
            # Split labels back per user
            cluster_labels = {}
            offset = 0
            for uid in selected_users:
                length = len(user_points[uid])
                cluster_labels[uid] = flat_labels[offset : offset + length]
                offset += length

        cluster_cmap = plt.colormaps['tab10']
        traj_colors = [plt.colormaps['hsv'](i / max(actual_n, 1)) for i in range(actual_n)]

        for i, user_id in enumerate(selected_users):
            rx_idxs = user_points[user_id]
            positions = self.rx_pos_all_masked[rx_idxs]

            if cluster_labels is not None:
                # Draw the connecting line in neutral grey, then scatter points
                # coloured by cluster
                point_colors = [
                    cluster_cmap(lbl / max(n_clusters - 1, 1)) for lbl in cluster_labels[user_id]
                ]
                for j in range(len(positions) - 1):
                    ax.plot(
                        positions[j : j + 2, 0],
                        positions[j : j + 2, 1],
                        '-',
                        color=point_colors[j],
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=3,
                    )
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    c=point_colors,
                    s=10,
                    linewidths=0,
                    alpha=0.85,
                    zorder=4,
                )
            else:
                color = traj_colors[i]
                if sequential_shading:
                    from matplotlib.collections import LineCollection

                    r, g, b, _ = color
                    n_pts = len(positions)
                    t_vals = np.linspace(0, 1, n_pts)
                    # light (white blend) → dark (full colour)
                    pt_colors = [
                        (1.0 - t * (1.0 - r), 1.0 - t * (1.0 - g), 1.0 - t * (1.0 - b), 0.85)
                        for t in t_vals
                    ]
                    segments = [positions[j : j + 2] for j in range(n_pts - 1)]
                    seg_colors = [pt_colors[j] for j in range(n_pts - 1)]
                    lc = LineCollection(segments, colors=seg_colors, linewidth=1.5, zorder=4)
                    ax.add_collection(lc)
                    ax.scatter(
                        positions[:, 0],
                        positions[:, 1],
                        c=pt_colors,
                        s=9,
                        linewidths=0,
                        zorder=4,
                    )
                    handle = ax.scatter([], [], color=color, label=f'Traj {user_id}', s=30)
                    legend_handles.append(handle)
                else:
                    (handle,) = ax.plot(
                        positions[:, 0],
                        positions[:, 1],
                        '-o',
                        color=color,
                        markersize=3,
                        linewidth=1.5,
                        label=f'Traj {user_id}',
                        alpha=0.85,
                        zorder=4,
                    )
                    legend_handles.append(handle)

            # Mark trajectory start
            if cluster_labels is not None:
                start_color = cluster_cmap(int(cluster_labels[user_id][0]) / max(n_clusters - 1, 1))
            elif sequential_shading:
                r, g, b, _ = traj_colors[i]
                start_color = (1.0, 1.0, 1.0, 0.85)  # lightest (t=0)
            else:
                start_color = traj_colors[i]
            ax.scatter(
                positions[0, 0],
                positions[0, 1],
                s=60,
                color=start_color,
                edgecolor='black',
                linewidth=0.6,
                zorder=6,
            )

        # Re-enforce limits — scene.plot() may have changed them
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_title(f'First {actual_n} trajectories — {stage} stage')
        # Use only our handles so scene labels (terrain, buildings) excluded
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7, ncol=2)
        plt.tight_layout()

        plot_name = (
            f'{self.cfg["scenario"]}_{stage}_trajectories.png' if plot_name is None else plot_name
        )
        plt.savefig(Path(f'imgs/{plot_name}'), dpi=300)
        plt.close()

        print(f'Trajectory plot saved to: {plot_name}')


if __name__ == '__main__':
    pass
