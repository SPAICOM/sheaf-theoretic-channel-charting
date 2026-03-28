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


class SharedTrajectoryCSIDataset(Dataset):
    """Dataset producing shared samples between base stations for CSI learning.

    This dataset provides paired CSI samples from overlapping coverage areas
    of two base stations, used for learning shared representations across
    the wireless network.

    Parameters
    ----------
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
    shared_pos : np.ndarray
        Positions of the trajectory anchor points in the shared coverage area.
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
        shared_pos: np.ndarray,
        channels_bs_1: np.ndarray,
        channels_bs_2: np.ndarray,
        idx_bs_1: int,
        idx_bs_2: int,
        shared_traj_idxs: np.ndarray | None = None,
    ):
        super().__init__()

        self.idx_bs_1 = idx_bs_1
        self.idx_bs_2 = idx_bs_2

        self.shared_pos = shared_pos
        self.shared_traj_idxs = shared_traj_idxs  # gidxs into rx_pos_all_masked

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
