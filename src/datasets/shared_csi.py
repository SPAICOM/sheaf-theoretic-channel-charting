from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def csi_to_realvec_ferrand(
    H,
    lag_step: int = 4,
    max_lag: int = 60,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Preprocess CSI following Ferrand et al. 'Triplet-Based Wireless Channel Charting'.

    Steps:
      1. 1D FFT over R (beam domain projection, replaces the paper's 2D FFT over n,m)
      2. Frequency-domain autocorrelation (expectation over f) at multiple lags
      3. Log-magnitude compression
      4. Flatten to real vector

    Parameters
    ----------
    H : torch.Tensor
        Complex CSI of shape (R, T, F).
        R : Rx antennas (flat index)
        T : Tx antennas (typically 1 for single UE)
        F : subcarriers
    lag_step : int
        Step between autocorrelation lags (default: 4).
    max_lag : int
        Maximum lag (default: 60).
    eps : float
        Numerical stability constant for log (default: 1e-12).

    Returns
    -------
    torch.Tensor
        Real feature vector of shape (R * T * num_lags,).
        With defaults: num_lags = len(range(0, 61, 4)) = 16.
    """
    device = H.device
    H_np = np.array(H)
    R, T, F = H_np.shape

    # Step 1: 1D FFT over R -> beam domain
    # (Replaces paper's 2D FFT over n,m since R is flat here)
    H_beam = np.fft.fft(H_np, axis=0)  # (R, T, F)

    # Step 2: Frequency autocorrelation at each lag delta
    # c(r, t, delta) = E_f[ H(r,t,f) * conj(H(r,t,f+delta)) ]
    lags = np.arange(0, max_lag + 1, lag_step)
    num_lags = len(lags)
    c = np.zeros((R, T, num_lags), dtype=np.complex128)

    for i, lag in enumerate(lags):
        if lag == 0:
            # Zero lag = mean power per (r, t)
            c[:, :, i] = np.mean(H_beam * np.conj(H_beam), axis=2)
        else:
            # Mean cross-correlation over valid frequency pairs
            c[:, :, i] = np.mean(
                H_beam[:, :, :-lag] * np.conj(H_beam[:, :, lag:]),
                axis=2,
            )

    # Step 3: Log of absolute value
    c = np.log(np.abs(c) + eps)  # (R, T, num_lags), real

    # Step 4: Flatten
    features = torch.from_numpy(c.reshape(-1).astype(np.float32)).to(device)
    return features


def csi_to_realvec_dichasus(
    H,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Preprocess CSI following the DICHASUS channel charting tutorial.

    Feature engineering based on subcarrier chunk averaging:
      1. Divide F subcarriers into non-overlapping chunks of size chunk_size
      2. Average H within each chunk -> reduces frequency dimension F -> F//chunk_size
      3. Stack real and imaginary parts -> real-valued output

    This is a simple but effective approach: neighboring subcarriers are highly
    correlated, so averaging reduces redundancy while preserving spatial information.
    No phase preprocessing is applied (assumes calibrated input or phase-robust NN).

    Parameters
    ----------
    H : torch.Tensor
        Complex CSI of shape (R, T, F).
        R : Rx antennas (flat index)
        T : Tx antennas (typically 1 for single UE)
        F : subcarriers (must be divisible by chunk_size)
    chunk_size : int
        Number of subcarriers per averaging chunk (default: 32).
        Output frequency dimension = F // chunk_size.

    Returns
    -------
    torch.Tensor
        Real feature vector of shape (R * T * (F // chunk_size) * 2,).
        The factor 2 comes from stacking real and imaginary parts.

    Example
    -------
    >>> H = torch.randn(32, 1, 288, dtype=torch.complex64)
    >>> features = csi_to_realvec_dichasus(H)
    >>> # shape: (32 * 1 * 9 * 2,) = (576,) for chunk_size=32
    """
    device = H.device
    H = np.array(H)
    R, T, F = H.shape

    assert F % chunk_size == 0, (
        f"F={F} must be divisible by chunk_size={chunk_size}"
    )
    num_chunks = F // chunk_size

    # Step 1+2: Average within each subcarrier chunk
    # H reshaped to (R, T, num_chunks, chunk_size), then averaged over last axis
    H_chunked = H.reshape(R, T, num_chunks, chunk_size)
    H_avg = H_chunked.mean(dim=-1)  # (R, T, num_chunks), complex

    # Step 3: Stack real and imaginary parts -> real tensor
    features = torch.stack([H_avg.real, H_avg.imag], dim=-1)  # (R, T, num_chunks, 2)

    # Step 4: Flatten
    features = features.reshape(-1).float().to(device)
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
        H_1 = csi_to_realvec_ferrand(self.channels_bs_1[index])
        H_2 = csi_to_realvec_ferrand(self.channels_bs_2[index])

        return H_1, H_2, (self.idx_bs_1, self.idx_bs_2)


if __name__ == '__main__':
    pass
