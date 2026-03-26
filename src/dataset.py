from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def csi_to_realvec(
    H: torch.Tensor,
    lag_step: int = 4,
    max_lag: int = 60,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Preprocess CSI tensor following 'Triplet-Based Wireless Channel Charting'.

    Converts complex-valued Channel State Information (CSI) tensors into
    real-valued feature vectors using 2D FFT and frequency autocorrelation.

    Parameters
    ----------
    H : torch.Tensor
        Complex CSI tensor of shape (R, T, F).
        R : receiver antennas
        T : transmitter antennas
        F : frequency subcarriers
    lag_step : int, optional
        Step between autocorrelation lags (default: 4).
    max_lag : int, optional
        Maximum lag for autocorrelation (default: 60).
    eps : float, optional
        Small constant for numerical stability in logarithm (default: 1e-12).

    Returns
    -------
    torch.Tensor
        Flattened real feature vector of shape (D,), where D is the
        total number of features after processing.

    Example
    -------
    >>> H = torch.randn(4, 2, 64, dtype=torch.complex64)  # (R, T, F)
    >>> features = csi_to_realvec(H)  # shape: (D,)
    """
    device = H.device
    H = np.array(H)
    R, T, F = H.shape

    # 2D FFT: transform from RT domain to beam angular domain
    H_beam = np.fft.fft2(H, axes=(0, 1))  # shape: (R, T, F)

    # Compute autocorrelation in frequency domain across lags
    lags = np.arange(0, max_lag + 1, lag_step)
    num_lags = len(lags)

    r = np.zeros((R, T, num_lags), dtype=np.complex128)

    for i, lag in enumerate(lags):
        if lag == 0:
            # Zero lag: autocorrelation is power sum across all frequencies
            r[:, :, i] = np.sum(H_beam * np.conj(H_beam), axis=2)
        else:
            # Non-zero lag: shifted correlation across frequency axis
            r[:, :, i] = np.sum(H_beam[:, :, :-lag] * np.conj(H_beam[:, :, lag:]), axis=2)

    # Log scaling for dynamic range compression
    r = np.log(np.abs(r) + eps)

    # Flatten to 1D feature vector and convert back to tensor
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
    """Dataset producing Siamese batches for CSI trajectory learning.

    Supports two sampling modes:

    - ``'triplet'``: returns (xA, xP, xN, y=-1) for triplet loss
    - ``'contrastive'``: returns (xA, xP, xN, y in {0,1}) for contrastive loss

    Positive samples are from the same user within a time window.
    Negative samples are from the same user outside the window.

    Parameters
    ----------
    idx_to_neg_pos : dict
        Mapping from (user_id, rx_idx) to dict with 'pos' and 'neg' keys
        containing lists of positive and negative indices.
    mask : np.ndarray
        Boolean mask indicating valid receiver positions.
    rx_pos : np.ndarray
        Positions of receiver antennas, shape (N_rx, 2 or 3).
    channels : np.ndarray
        Complex CSI data for each receiver position, shape (N_rx, R, T, F).
    pair_mode : str, optional
        Sampling mode: ``'triplet'`` or ``'contrastive'`` (default: ``'triplet'``).
    p_positive : float, optional
        Probability of sampling a positive pair in contrastive mode
        (default: 0.5). Only used when ``pair_mode='contrastive'``.

    Attributes
    ----------
    pair_mode : str
        The current sampling mode.
    p_positive : float
        Probability of positive sampling in contrastive mode.

    Notes
    -----
    - Complex CSI is converted to real-valued vectors via :func:`csi_to_realvec`.
    - Trajectories are built at initialization and stored in ``idx_to_neg_pos``.
    - Uses a random number generator for stochastic sampling in contrastive mode.
    """

    def __init__(
        self,
        idx_to_neg_pos: dict[tuple[int, int], dict[str, list[int]]],
        mask: np.ndarray,
        rx_pos: np.ndarray,
        channels: np.ndarray,
        pair_mode: str = 'triplet',
        p_positive: float = 0.5,
    ):
        super().__init__()

        # Validate sampling mode
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.p_positive = float(p_positive)

        self.channels = channels
        # Initialize random generator for contrastive sampling
        self.rng = np.random.default_rng()

        # Build valid index set for filtering trajectories
        self.idx_to_neg_pos = idx_to_neg_pos.copy()
        self.valid_idxs = np.where(mask)[0]
        for user_id, idx in list(self.idx_to_neg_pos.keys()):
            if idx not in self.valid_idxs:
                del self.idx_to_neg_pos[(user_id, idx)]

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
                xP = torch.vstack([csi_to_realvec(self._H_from_global_index(i)) for i in pos_idxs])
                xN = torch.vstack([csi_to_realvec(self._H_from_global_index(i)) for i in neg_idxs])

                y = torch.tensor(-1, dtype=torch.long)  # <-- IMPORTANT

            case 'contrastive':
                # Contrastive:
                # sample positive pair with prob p_positive else negative pair
                if self.rng.random() < self.p_positive:
                    xA = csi_to_realvec(H_A)
                    xP = torch.vstack(
                        [csi_to_realvec(self._H_from_global_index(i)) for i in pos_idxs]
                    )
                    xN = torch.vstack(
                        [csi_to_realvec(self._H_from_global_index(i)) for i in neg_idxs]
                    )
                    y = torch.tensor(1, dtype=torch.long)
                else:
                    xA = csi_to_realvec(H_A)
                    xP = torch.vstack(
                        [csi_to_realvec(self._H_from_global_index(i)) for i in pos_idxs]
                    )
                    xN = torch.vstack(
                        [csi_to_realvec(self._H_from_global_index(i)) for i in neg_idxs]
                    )
                    y = torch.tensor(0, dtype=torch.long)

        return xA, xP, xN, y

        # xA: [BS, d]
        # xP: [BS, n_pos, d]
        # xN: [BS, n_neg, d]


class SharedTrajectoryCSIDataset(Dataset):
    """Dataset producing shared samples between base stations for CSI learning.

    This dataset provides paired CSI samples from overlapping coverage areas
    of two base stations, used for learning shared representations across
    the wireless network.

    Parameters
    ----------
    shared_mask : np.ndarray
        Boolean mask indicating valid shared positions.
    shared_pos : np.ndarray
        Positions of shared receiver locations, shape (N_shared, 2 or 3).
    channels_bs_1 : np.ndarray
        Complex CSI data from base station 1, shape (N_shared, R, T, F).
    channels_bs_2 : np.ndarray
        Complex CSI data from base station 2, shape (N_shared, R, T, F).
    idx_bs_1 : int
        Index identifier for base station 1.
    idx_bs_2 : int
        Index identifier for base station 2.

    Attributes
    ----------
    idx_bs_1 : int
        Index of the first base station.
    idx_bs_2 : int
        Index of the second base station.
    channels_bs_1 : np.ndarray
        CSI data from base station 1.
    channels_bs_2 : np.ndarray
        CSI data from base station 2.

    Notes
    -----
    - Complex CSI is converted to real-valued vectors via :func:`csi_to_realvec`.
    - Returns tuple (H_1, H_2, (idx_bs_1, idx_bs_2)) for downstream processing.
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

        # Store CSI data: (num_points, R, T, F) for each base station
        self.channels_bs_1 = channels_bs_1
        self.channels_bs_2 = channels_bs_2

    def __len__(self) -> int:
        """Return the total number of shared samples.

        Returns
        -------
        int
            Number of shared receiver positions between the two base stations.
        """
        return self.channels_bs_1.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Return a shared sample pair from both base stations.

        Parameters
        ----------
        index : int
            Index into the shared dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
            Tuple containing:
            - H_1: Real-valued CSI vector from BS1
            - H_2: Real-valued CSI vector from BS2
            - (idx_bs_1, idx_bs_2): Base station indices for identification
        """
        H_1 = csi_to_realvec(self.channels_bs_1[index])
        H_2 = csi_to_realvec(self.channels_bs_2[index])

        return H_1, H_2, (self.idx_bs_1, self.idx_bs_2)


if __name__ == '__main__':
    pass
