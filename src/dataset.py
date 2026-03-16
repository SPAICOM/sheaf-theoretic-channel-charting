import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import Dataset


def csi_to_realvec(
    H: torch.Tensor, lag_step: int = 4, max_lag: int = 60, eps: float = 1e-12
):
    """
    Preprocess CSI tensor following the pipeline in
    'Triplet-Based Wireless Channel Charting'.

    Parameters
    ----------
    H : torch.Tensor
        Complex CSI tensor of shape (R, T, F)
        R = receiver antennas
        T = transmitter antennas
        F = frequency subcarriers

    lag_step : int
        Step between autocorrelation lags (default 4)

    max_lag : int
        Maximum lag (default 60)

    eps : float
        Small constant for numerical stability in log

    Returns
    -------
    features : np.ndarray
        Flattened real feature vector
    """
    device = H.device
    H = H.detach().cpu().numpy()
    R, T, F = H.shape

    # 2 dimensional FT (from RT domain to beam angular domain)
    H_beam = np.fft.fft2(H, axes=(0, 1))  # shape: (R, T, F)

    # Autocorrelation in frequency
    lags = np.arange(0, max_lag + 1, lag_step)
    num_lags = len(lags)

    r = np.zeros((R, T, num_lags), dtype=np.complex128)

    for i, lag in enumerate(lags):
        if lag == 0:
            r[:, :, i] = np.sum(H_beam * np.conj(H_beam), axis=2)
        else:
            r[:, :, i] = np.sum(
                H_beam[:, :, :-lag] * np.conj(H_beam[:, :, lag:]), axis=2
            )

    # Log scaling
    r = np.log(np.abs(r) + eps)

    # Flatten to vector
    features = r.reshape(-1)
    features = torch.from_numpy(features).to(device)

    return features


# def csi_to_realvec(
#     H: torch.Tensor,
#     c: float = 1e7,
# ) -> torch.Tensor:
#     """
#     Convert a complex CSI tensor into a real-valued vector.

#     Parameters
#     ----------
#     H : torch.Tensor
#         Complex-valued CSI tensor, shape (...).

#     Returns
#     -------
#     torch.Tensor
#         Real-valued vector, shape (D,), where D = 2 * product of H dims
#         except the first (sample) dimension.
#     """
#     # Flatten complex tensor into real vector (real + imag)
#     H = H * torch.tensor(c)
#     x = torch.view_as_real(H).reshape(-1).float()
#     return x


class TrajectoryCSIDataset(Dataset):
    """
    Dataset producing Siamese batches for CSI trajectory learning.

    Supports two sampling modes:

    - 'triplet': returns (xA, xP, xN, y=-1)
    - 'contrastive': returns (xA, xP, xN=None, y in {0,1})

    Positives:
        Same user, |dt| <= window, excluding anchor.
    Negatives:
        Same user outside window if

    Parameters
    ----------
    rx_pos : np.ndarray
        Positions of receiver antennas, shape (N_rx, 2 or 3).
    H_users : np.ndarray
        CSI data per RX position, shape (N_rx, ...).
    num_users : int, optional
        Number of simulated users (default: 500).
    T_min : int, optional
        Minimum trajectory length (default: 32).
    T_max : int, optional
        Maximum trajectory length (default: 128).
    kinds : tuple, optional
        Trajectory types: 'linear', 'circular', 'random' (default all).
    pair_mode : str, optional
        'triplet' or 'contrastive' (default: 'triplet').
    window : int, optional
        Time window for positive sampling (default: 3).
    p_positive : float, optional
        Probability of positive pair in contrastive mode (default: 0.5).
    seed : int, optional
        Random seed (default: 0).

    Notes
    -----
    - Converts complex CSI to real-valued vectors with `csi_to_realvec`.
    - Builds variable-length trajectories per user at initialization.
    - Provides methods to sample positives and negatives efficiently.
    """

    def __init__(
        self,
        idx_to_neg_pos: dict,
        mask: np.ndarray,
        rx_pos: np.ndarray,
        H_users: np.ndarray,
        # bs_pos: np.ndarray = None,
        # num_users: int = 1,
        # T_min: int = 32,
        # T_max: int = 128,
        # trajectory_kind: str | None = None,
        # linear_len=(20.0, 120.0),
        # circle_r=(10.0, 60.0),
        # random_step=(1.0, 5.0),
        # random_keep_dir=0.7,
        # z_min: float | None = None,
        # z_max: float | None = None,
        # r_min: float | None = None,
        # r_max: float | None = None,
        # coverage_area: float = 0.2,
        # bias_sampling: bool = False,
        # seed: int = 0,
        # # --- Siamese sampling controls ---
        # pair_mode: str = 'triplet',  # "triplet" or "contrastive"
        # in_window: int = 3,
        # out_window: int = 6,
        # p_positive: float = 0.5,  # only for contrastive
    ):
        super().__init__()

        # Siamese sampling configuration
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.in_window = int(in_window)
        self.out_window = int(out_window)
        assert out_window > in_window, (
            '"out_window" must always be grater than "in_window"'
        )
        self.p_positive = float(p_positive)
        self.bias_sampling = bool(bias_sampling)

        self.rng = np.random.default_rng(seed)

        # Convert RX positions to array, ensure 3D coordinates
        rx_pos = np.asarray(rx_pos, dtype=np.float64)
        if rx_pos.shape[1] == 2:
            rx_pos = np.c_[rx_pos, np.zeros((rx_pos.shape[0], 1))]
        self.rx_pos_all = rx_pos

        # Convert CSI to array and validate dimensions
        H_users = np.asarray(H_users)
        if H_users.shape[0] != rx_pos.shape[0]:
            raise ValueError('H_users first dim must match rx_pos first dim')
        self.H_users = H_users

        # Filter candidate RX points (optional)
        mask = np.ones(len(rx_pos), dtype=bool)
        if z_min is not None:
            mask &= rx_pos[:, 2] >= float(z_min)
        if z_max is not None:
            mask &= rx_pos[:, 2] <= float(z_max)

        # distance-to-BS filtering
        bs_pos = np.asarray(bs_pos)
        if bs_pos.shape[0] == 2:
            bs_pos = np.r_[bs_pos, 0.0]

        d = np.linalg.norm(rx_pos - bs_pos, axis=1)

        if r_min is not None:
            mask &= d >= float(r_min)

        assert coverage_area >= 0 and coverage_area <= 1, (
            '"coverage_area" must be between 0 and 1 for a BS.'
        )
        if r_max is None:
            xmin, ymin = rx_pos[:, :2].min(axis=0)
            xmax, ymax = rx_pos[:, :2].max(axis=0)
            r_max = coverage_area * min(xmax - xmin, ymax - ymin)

        mask &= d <= float(r_max)

        self.valid_idxs = np.where(mask)[0]
        if len(self.valid_idxs) < 10:
            raise ValueError('Too few valid RX points after filtering.')
        self.rx_pos = rx_pos[self.valid_idxs]
        self.rx_xy = self.rx_pos[:, :2]
        self.kdtree = KDTree(self.rx_xy)

        self.num_users = int(num_users)
        self.T_min = int(T_min)
        self.T_max = int(T_max)
        if self.T_min < 2 or self.T_max < self.T_min:
            raise ValueError('Bad T_min/T_max')
        self.kinds = ('linear', 'circular', 'random')
        self.trajectory_kind = trajectory_kind
        assert (self.trajectory_kind in self.kinds) or (
            self.trajectory_kind is None
        ), (
            f'Trajectory kind "{self.trajectory_kind}" not available,'
            + 'possible values: ["linear", "circular", "random", None]'
        )

        self.linear_len = linear_len
        self.circle_r = circle_r
        self.random_step = random_step
        self.random_keep_dir = float(random_keep_dir)

        # ---- Build variable-length trajectories once ----
        self.idx_to_neg_pos = idx_to_neg_pos
        self.valid_idxs = np.where(mask)[0]
        for user_id, idx in self.idx_to_neg_pos.keys():
            if idx not in self.valid_idxs:
                del self.idx_to_neg_pos[(user_id, idx)]

        # for user_id in range(self.num_users):
        #     # Random trajectory length
        #     T = int(self.rng.integers(self.T_min, self.T_max + 1))

        #     # Randomly pick trajectory kind
        #     kind = (
        #         self.kinds[int(self.rng.integers(0, len(self.kinds)))]
        #         if self.trajectory_kind is None
        #         else self.trajectory_kind
        #     )

        #     # Generate trajectory of RX indices
        #     rx_idxs = self._generate_one(kind, T)
        #     max_idx = np.max(rx_idxs)
        #     min_idx = np.min(rx_idxs)
        #     for idx in rx_idxs:
        #         min_point = min_idx + self.out_window
        #         max_point = max_idx - self.out_window
        #         if idx > min_point and idx < max_point:
        #             pos = np.clip(
        #                 np.arange(
        #                     idx - self.in_window, idx + self.in_window + 1
        #                 ),
        #                 a_min=0,
        #                 a_max=max_idx,
        #             )
        #             pos = pos[pos != idx]
        #             neg = np.clip(
        #                 np.concat(
        #                     [
        #                         np.arange(
        #                             idx - self.out_window, idx - self.in_window
        #                         ),
        #                         np.arange(
        #                             idx + self.in_window + 1,
        #                             idx + self.out_window + 1,
        #                         ),
        #                     ]
        #                 ),
        #                 a_min=0,
        #                 a_max=max_idx,
        #             )
        #             neg = neg[neg != idx]
        #             self.idx_to_neg_pos[(user_id, idx)] = {
        #                 'pos': pos,
        #                 'neg': neg,
        #             }

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
            idx = self.rng.choice(len(self.rx_xy), p=prob)
        else:
            idx = int(self.rng.integers(0, len(self.rx_xy)))
        return self.rx_xy[idx].copy()

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
                xy[t] = xy[t - 1] + step * d
                prev_dir = d
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

    # ----------------- Dataset API -----------------
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of trajectory points across all users.
        """
        return len(self.idx_to_neg_pos)

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one Siamese sample depending on pair_mode.

        Returns
        -------
        xA, xP, xN, y
            CSI vectors and label / placeholder.
        """
        # Anchor
        id = list(self.idx_to_neg_pos.keys())[index]
        H_A = self._H_from_global_index(id[1])

        pos_idxs = self.idx_to_neg_pos[id]['pos']
        neg_idxs = self.idx_to_neg_pos[id]['neg']

        match self.pair_mode:
            case 'triplet':
                # Triplet

                xA = csi_to_realvec(H_A)
                xP = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i))
                        for i in pos_idxs
                    ]
                )
                xN = torch.vstack(
                    [
                        csi_to_realvec(self._H_from_global_index(i))
                        for i in neg_idxs
                    ]
                )

                y = torch.tensor(-1, dtype=torch.long)  # <-- IMPORTANT

            case 'contrastive':
                # Contrastive:
                # sample positive pair with prob p_positive else negative pair
                if self.rng.random() < self.p_positive:
                    xA = csi_to_realvec(H_A)
                    xP = torch.vstack(
                        [
                            csi_to_realvec(self._H_from_global_index(i))
                            for i in pos_idxs
                        ]
                    )
                    xN = torch.vstack(
                        [
                            csi_to_realvec(self._H_from_global_index(i))
                            for i in neg_idxs
                        ]
                    )
                    y = torch.tensor(1, dtype=torch.long)
                else:
                    xA = csi_to_realvec(H_A)
                    xP = torch.vstack(
                        [
                            csi_to_realvec(self._H_from_global_index(i))
                            for i in pos_idxs
                        ]
                    )
                    xN = torch.vstack(
                        [
                            csi_to_realvec(self._H_from_global_index(i))
                            for i in neg_idxs
                        ]
                    )
                    y = torch.tensor(0, dtype=torch.long)

        return xA, xP, xN, y

        # xA: [BS, d]
        # xP: [BS, n_pos, d]
        # xN: [BS, n_neg, d]


class SharedTrajectoryCSIDataset(Dataset):
    """
    Dataset producing shared batches between bss for CSI trajectory learning.
    """

    def __init__(
        self,
        shared_mask: np.ndarray,
        shared_pos: np.ndarray,
        channels_bs_1: np.ndarray,
        channels_bs_2: np.ndarray,
    ):
        super().__init__()

        self.idx_bs_1 = idx_bs_1
        self.idx_bs_2 = idx_bs_2

        self.shared_mask = shared_mask
        self.shared_pos = shared_pos

        self.channels_bs_1 = channels_bs_1  # (num_points, R, T, F)
        self.channels_bs_2 = channels_bs_2  # (num_points, R, T, F)

    # ----------------- Dataset API -----------------
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of trajectory points across all users.
        """
        return self.channels_bs_1.shape[0]

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one Siamese sample depending on pair_mode.

        Returns
        -------
        xA, xP, xN, y
            CSI vectors and label / placeholder.
        """
        # Anchor retrieving
        H_1 = csi_to_realvec(self.channels_bs_1[index])
        H_2 = csi_to_realvec(self.channels_bs_2[index])

        return H_1, H_2, (self.idx_bs_1, self.idx_bs_2)
