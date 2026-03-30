from __future__ import annotations

import numpy as np
import torch


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
    H_beam = np.fft.fft(H_np, axis=0)  # (R, T, F)

    # Step 2: Frequency autocorrelation at each lag delta
    lags = np.arange(0, max_lag + 1, lag_step)
    num_lags = len(lags)
    c = np.zeros((R, T, num_lags), dtype=np.complex128)

    for i, lag in enumerate(lags):
        if lag == 0:
            c[:, :, i] = np.mean(H_beam * np.conj(H_beam), axis=2)
        else:
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
    H: torch.Tensor,
    chunk_size: int = 32,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Preprocess CSI following the DICHASUS channel charting tutorial.

    Feature engineering based on subcarrier chunk averaging:
      1. Divide F subcarriers into non-overlapping chunks of size chunk_size
      2. Average H within each chunk -> reduces frequency dimension F -> F//chunk_size
      3. Element-wise normalization (divide by average power)
      4. Stack real and imaginary parts -> real-valued output

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
    normalize : bool
        Whether to apply element-wise normalization (default: True).
        Normalizes by dividing each element by the average power across all
        antennas, tx, and frequency bins.
    eps : float
        Numerical stability constant for normalization (default: 1e-12).

    Returns
    -------
    torch.Tensor
        Real feature vector of shape (R * T * (F // chunk_size) * 2,).
        The factor 2 comes from stacking real and imaginary parts.
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H)
    device = H.device
    R, T, F = H.shape

    assert F % chunk_size == 0, f'F={F} must be divisible by chunk_size={chunk_size}'
    num_chunks = F // chunk_size

    H_chunked = H.reshape(R, T, num_chunks, chunk_size)
    H_avg = H_chunked.mean(dim=-1)

    if normalize:
        power = torch.mean(torch.abs(H_avg) ** 2)
        H_avg = H_avg / (torch.sqrt(power) + eps)

    features = torch.stack([H_avg.real, H_avg.imag], dim=-1)
    features = features.reshape(-1).float().to(device)
    return features


def csi_to_realvec(
    H,
    method: str = 'ferrand',
    lag_step: int = 4,
    max_lag: int = 60,
    chunk_size: int = 32,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Wrapper for CSI to real vector conversion.

    Parameters
    ----------
    H : torch.Tensor
        Complex CSI of shape (R, T, F).
    method : str
        Conversion method: 'ferrand' (default) or 'dichasus'.
    lag_step : int
        Only for 'ferrand' method. Step between autocorrelation lags.
    max_lag : int
        Only for 'ferrand' method. Maximum lag.
    chunk_size : int
        Only for 'dichasus' method. Number of subcarriers per averaging chunk.
    normalize : bool
        Only for 'dichasus' method. Whether to apply element-wise normalization.
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Real feature vector.
    """
    if method == 'ferrand':
        return csi_to_realvec_ferrand(H, lag_step, max_lag, eps)
    elif method == 'dichasus':
        return csi_to_realvec_dichasus(H, chunk_size, normalize, eps)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ferrand' or 'dichasus'.")


if __name__ == '__main__':
    pass
