import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union

def compute_xi(DD: npt.NDArray[np.float64],
               DR: npt.NDArray[np.float64],
               RR: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute the 2-point correlation function ξ(s, μ) using the Landy-Szalay estimator.

    The estimator is defined as:
        ξ = (DD - 2 DR + RR) / RR
    where DD, DR, and RR are the data-data, data-random, and random-random
    pair counts, respectively.

    Parameters
    ----------
    DD : np.ndarray
        Data-data pair counts, can be a 1D or 2D array depending on the
        (s, μ) binning.
    DR : np.ndarray
        Data-random pair counts, same shape as DD.
    RR : np.ndarray
        Random-random pair counts, same shape as DD.

    Returns
    -------
    np.ndarray
        Correlation function ξ(s, μ) evaluated for each bin. Same shape as
        input arrays. Bins with RR=0 are set to zero to avoid division by zero.

    Notes
    -----
    - Inputs are automatically converted to floating point arrays to ensure
      correct division.
    - The function handles 1D or flattened 2D arrays consistently.
    """

    # ensure floating point division
    DD = DD.astype(float)
    DR = DR.astype(float)
    RR = RR.astype(float)

    # avoid division by zero
    mask = RR != 0
    xi = np.zeros_like(DD)

    xi[mask] = (DD[mask] - 2*DR[mask] + RR[mask]) / RR[mask]

    return xi

def rebin_SMU(s_matrix: npt.NDArray[np.float64],
              mu_matrix: npt.NDArray[np.float64],
              dd_matrix: npt.NDArray[np.float64],
              dr_matrix: npt.NDArray[np.float64],
              rr_matrix: npt.NDArray[np.float64],
              delta_s: int = 5
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Rebin the pair-count matrices (DD, DR, RR) in the (s, μ) coordinate space by
    grouping `delta_s` consecutive separation bins while keeping the μ-binning unchanged.
    This operation reduces the number of s-bins and constructs flattened arrays for
    subsequent SMU analyses.

    Parameters
    ----------
    s_matrix : np.ndarray
        Array of shape `(n_files, ns * nmu)` containing the separation values `s`
        associated with each pair count, flattened row-wise over the μ-bins.
    mu_matrix : np.ndarray
        Array of shape `(n_files, ns * nmu)` containing the cosine of the angle μ
        for each (s, μ) cell, flattened consistently with `s_matrix`.
    dd_matrix : np.ndarray
        Array of shape `(n_files, ns * nmu)` containing the data-data (DD) pair counts.
    dr_matrix : np.ndarray
        Array of shape `(n_files, ns * nmu)` containing the data-random (DR) pair counts.
    rr_matrix : np.ndarray
        Array of shape `(n_files, ns * nmu)` containing the random-random (RR) pair counts.
    delta_s : int, optional
        Number of consecutive separation bins to merge into a single re-binned s-bin.
        Default is `5` Mpc/h, corresponding to 40 bins in s.

    Returns
    -------
    tuple
        `(s_flat, mu_flat, dd_flat, dr_flat, rr_flat)` where:
        - `s_flat` : np.ndarray  
          Re-binned separation values for each file, flattened over μ.  
          Shape: `(n_files, new_ns * nmu)`.
        - `mu_flat` : np.ndarray  
          μ-values replicated for each new s-bin, flattened over s.  
          Shape: `(n_files, new_ns * nmu)`.
        - `dd_flat` : np.ndarray  
          Re-binned DD counts obtained by summing over each block of `delta_s`  
          consecutive s-bins.  
          Shape: `(n_files, new_ns * nmu)`.
        - `dr_flat` : np.ndarray  
          Re-binned DR counts, shaped as above.
        - `rr_flat` : np.ndarray  
          Re-binned RR counts, shaped as above.

          Here `new_ns = ns // delta_s`.

    Notes
    -----
    - The re-binning is performed only along the separation axis `s`; μ-bin structure
      is preserved.
    - The new separation values correspond to the mean of each `delta_s`-bin block.
    """

    ns = len(np.unique(s_matrix[0]))
    nmu = len(np.unique(mu_matrix[0]))
    new_ns = ns // delta_s

    n_files = dd_matrix.shape[0]

    dd_flat = np.zeros((n_files, new_ns * nmu))
    dr_flat = np.zeros((n_files, new_ns * nmu))
    rr_flat = np.zeros((n_files, new_ns * nmu))

    s_flat = np.zeros((n_files, new_ns * nmu))
    mu_flat = np.zeros((n_files, new_ns * nmu))

    for f in range(n_files):

        # reshape in order to sum easily
        s_vals = s_matrix[f].reshape(ns, nmu)[:, 0]
        mu_vals = mu_matrix[f].reshape(ns, nmu)[0, :]
        DD = dd_matrix[f].reshape(ns, nmu)
        DR = dr_matrix[f].reshape(ns, nmu)
        RR = rr_matrix[f].reshape(ns, nmu)

        # rebinning by summing in s
        DD_reb = DD.reshape(new_ns, delta_s, nmu).sum(axis=1)
        DR_reb = DR.reshape(new_ns, delta_s, nmu).sum(axis=1)
        RR_reb = RR.reshape(new_ns, delta_s, nmu).sum(axis=1)

        # flatten into 1D-array
        dd_flat[f] = DD_reb.reshape(new_ns * nmu)
        dr_flat[f] = DR_reb.reshape(new_ns * nmu)
        rr_flat[f] = RR_reb.reshape(new_ns * nmu)

        # new s (mean of each delta_s block)
        s_new_vals = np.array([
            s_vals[i*delta_s:(i+1)*delta_s].mean()
            for i in range(new_ns)
        ])

        # build flattened s-array for this file
        s_flat[f] = np.repeat(s_new_vals, nmu)

        # mu stays the same but must be repeated for each s-row
        mu_flat[f] = np.tile(mu_vals, new_ns)

    return s_flat, mu_flat, dd_flat, dr_flat, rr_flat


def rebin_RpPI(rp_matrix: npt.NDArray[np.float64],
               pi_matrix: npt.NDArray[np.float64],
               dd_matrix: npt.NDArray[np.float64],
               dr_matrix: npt.NDArray[np.float64],
               rr_matrix: npt.NDArray[np.float64],
               delta_rp: int = 5,
               delta_pi: int = 5
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Rebin the pair-count matrices (DD, DR, RR) in the (rₚ, π) coordinate space
    by aggregating blocks of size (`delta_rp`, `delta_pi`).
    This operation reduces the number of both rₚ-bins and π-bins and constructs
    flattened arrays for subsequent RpPI analyses.

    Parameters
    ----------
    rp_matrix : np.ndarray
        Array of shape `(n_files, nrp * npi)` containing the transverse
        separation values rₚ for each file arranged on a regular grid.
    pi_matrix : np.ndarray
        Array of shape `(n_files, nrp * npi)` containing the line-of-sight
        separations π for each file, aligned with `rp_matrix`.
    dd_matrix : np.ndarray
        Data-data pair counts on the (rₚ, π) grid. Shape `(n_files, nrp * npi)`.
    dr_matrix : np.ndarray
        Data-random pair counts, same shape as `dd_matrix`.
    rr_matrix : np.ndarray
        Random-random pair counts, same shape as `dd_matrix`.
    delta_rp : int, optional
        Number of consecutive separation bins to merge into a single re-binned rₚ-bin.
        Default is `5` Mpc/h, corresponding to 40 bins in rₚ.
    delta_pi : int, optional
        Number of consecutive separation bins to merge into a single re-binned π-bin.
        Default is `5` Mpc/h, corresponding to 40 bins in π.

    Returns
    -------
    tuple
        `(rp_flat, pi_flat, dd_flat, dr_flat, rr_flat)` where:
        - `rp_flat` : np.ndarray  
          Re-binned transverse separation values for each file.
          Shape: `(n_files, new_nrp * new_npi)`.
        - `mu_flat` : np.ndarray  
          Re-binned line-of-sight separation values for each file.
          Shape: `(n_files, new_nrp * new_npi)`.
        - `dd_flat` : np.ndarray  
          Re-binned DD counts obtained by summing over each block of `delta_rp`
          consecutive rp-bins and `delta_pi` consecutive pi-bins.
          Shape: `(n_files, new_nrp * new_npi)`.
        - `dr_flat` : np.ndarray  
          Re-binned DR counts, shaped as above.
        - `rr_flat` : np.ndarray  
          Re-binned RR counts, shaped as above.

          Here `new_nrp = nrp // delta_rp` and `new_npi = npi // delta_pi`.

    Notes
    -----
    - Rebinning is performed by summing the counts over non-overlapping blocks
      of size (`delta_rp`, `delta_pi`).
    - The rebinned coordinate values are computed as the mean of each block.
    """

    n_files = dd_matrix.shape[0]

    # infer original grid size
    nrp = len(np.unique(rp_matrix[0]))
    npi = len(np.unique(pi_matrix[0]))

    new_nrp = nrp // delta_rp
    new_npi = npi // delta_pi

    # output arrays
    rp_flat = np.zeros((n_files, new_nrp * new_npi))
    pi_flat = np.zeros((n_files, new_nrp * new_npi))
    dd_flat = np.zeros((n_files, new_nrp * new_npi))
    dr_flat = np.zeros((n_files, new_nrp * new_npi))
    rr_flat = np.zeros((n_files, new_nrp * new_npi))

    for f in range(n_files):

        # reshape coordinates and pair counts
        rp_vals = rp_matrix[f].reshape(nrp, npi)[:, 0]
        pi_vals = pi_matrix[f].reshape(nrp, npi)[0, :]

        DD = dd_matrix[f].reshape(nrp, npi)
        DR = dr_matrix[f].reshape(nrp, npi)
        RR = rr_matrix[f].reshape(nrp, npi)

        # rebinning in both r_p and pi
        DD_reb = DD.reshape(new_nrp, delta_rp, new_npi, delta_pi).sum(axis=(1, 3))
        DR_reb = DR.reshape(new_nrp, delta_rp, new_npi, delta_pi).sum(axis=(1, 3))
        RR_reb = RR.reshape(new_nrp, delta_rp, new_npi, delta_pi).sum(axis=(1, 3))

        # flatten into 1D-array
        dd_flat[f] = DD_reb.reshape(new_nrp * new_npi)
        dr_flat[f] = DR_reb.reshape(new_nrp * new_npi)
        rr_flat[f] = RR_reb.reshape(new_nrp * new_npi)

        # New coordinates: mean for each rebinned block
        rp_new_vals = np.array([
            rp_vals[i*delta_rp:(i+1)*delta_rp].mean()
            for i in range(new_nrp)
        ])
        pi_new_vals = np.array([
            pi_vals[j*delta_pi:(j+1)*delta_pi].mean()
            for j in range(new_npi)
        ])

        # flatten coordinates
        rp_flat[f] = np.repeat(rp_new_vals, new_npi)
        pi_flat[f] = np.tile(pi_new_vals, new_nrp)

    return rp_flat, pi_flat, dd_flat, dr_flat, rr_flat