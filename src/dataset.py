import numpy as np
import torch
from torch.utils.data import Dataset


def csi_to_realvec(
    H,
    lag_step: int = 4,
    max_lag: int = 60,
    eps: float = 1e-12,
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
    # H = torch.tensor(H).detach().clone()
    H = np.array(H)
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
    features = torch.from_numpy(features).float().to(device)

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
#     print(H.shape)
#     # print(H)
#     print(torch.tensor(H))
#     H = torch.tensor(H) * torch.tensor(c)
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
        channels: np.ndarray,
        pair_mode: str = 'triplet',
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
        p_positive: float = 0.5,  # only for contrastive
    ):
        super().__init__()

        # Siamese sampling configuration
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.p_positive = float(p_positive)

        self.channels = channels

        # ---- Build variable-length trajectories once ----
        self.idx_to_neg_pos = idx_to_neg_pos.copy()
        self.valid_idxs = np.where(mask)[0]
        for user_id, idx in idx_to_neg_pos:
            if idx not in self.valid_idxs:
                del self.idx_to_neg_pos[(user_id, idx)]

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
        return torch.from_numpy(self.channels[gidx])  # complex tensor

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
        idx_bs_1: int,
        idx_bs_2: int,
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


if __name__ == '__main__':
    pass
