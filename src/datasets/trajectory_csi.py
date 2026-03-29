from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .shared_csi import csi_to_realvec_ferrand, csi_to_realvec_dichasus


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
    valid_idxs : np.ndarray
        Valid indexes for the specific base station.
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
        valid_idxs: np.ndarray,
        rx_pos: np.ndarray,
        channels: np.ndarray,
        pair_mode: str = 'triplet',
        p_positive: float = 0.5,
        return_raw: bool = False,
    ):
        super().__init__()

        # Validate sampling mode
        assert pair_mode in ('triplet', 'contrastive')
        self.pair_mode = pair_mode
        self.p_positive = float(p_positive)

        self.channels = channels
        self.return_raw = return_raw
        # Initialize random generator for contrastive sampling
        self.rng = np.random.default_rng()

        # Build valid index set for filtering trajectories
        self.idx_to_neg_pos = idx_to_neg_pos.copy()
        self.valid_idxs = valid_idxs
        for user_id, idx in idx_to_neg_pos:
            if idx not in self.valid_idxs:
                del self.idx_to_neg_pos[(user_id, idx)]

        self.rx_pos = rx_pos

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

    def _pos_from_global_index(
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
        return torch.from_numpy(self.rx_pos[gidx])  # complex tensor

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

        If ``return_raw=True`` the method skips feature extraction and returns
        ``(H_A, gidx, pos)`` — the raw complex CSI tensor, the global masked
        index, and the 2-D position — which is useful for sanity-checking the
        index mapping against the original DeepMIMO arrays.

        Returns
        -------
        xA, xP, xN, y, pos  (normal mode)
            Processed CSI feature vectors, label, and anchor position.
        H_A, gidx, pos  (return_raw=True)
            Raw complex CSI tensor, masked global index, and anchor position.
        """
        # Anchor
        id = list(self.idx_to_neg_pos.keys())[index]
        gidx = id[1]
        H_A = self._H_from_global_index(gidx)
        pos = self._pos_from_global_index(gidx)

        if self.return_raw:
            return H_A, gidx, pos

        pos_idxs = self.idx_to_neg_pos[id]['pos']
        neg_idxs = self.idx_to_neg_pos[id]['neg']

        match self.pair_mode:
            case 'triplet':
                xA = csi_to_realvec_ferrand(H_A)
                xP = torch.vstack([csi_to_realvec_ferrand(self._H_from_global_index(i)) for i in pos_idxs])
                xN = torch.vstack([csi_to_realvec_ferrand(self._H_from_global_index(i)) for i in neg_idxs])
                y = torch.tensor(-1, dtype=torch.long)

            case 'contrastive':
                # sample positive pair with prob p_positive else negative pair
                xA = csi_to_realvec_ferrand(H_A)
                xP = torch.vstack([csi_to_realvec_ferrand(self._H_from_global_index(i)) for i in pos_idxs])
                xN = torch.vstack([csi_to_realvec_ferrand(self._H_from_global_index(i)) for i in neg_idxs])
                y = torch.tensor(
                    1 if self.rng.random() < self.p_positive else 0,
                    dtype=torch.long,
                )

        return xA, xP, xN, y, pos

        # xA: [BS, d]
        # xP: [BS, n_pos, d]
        # xN: [BS, n_neg, d]
