import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = True  # use real LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

try:
    matplotlib.use("QtAgg")
except ImportError:
    print("QtAgg backend not available, using default backend.")
matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


def plot_contourf(coords: str,
                  x_array: npt.NDArray[np.float64],
                  y_array: npt.NDArray[np.float64],
                  z_array: npt.NDArray[np.float64],
                  base_path: str,
                  kind: str,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  v_min: Optional[float] = None,
                  v_max: Optional[float] = None,
                  lvls: int = 20,
                  draw_lines: bool = False
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Generate a filled contour plot of the correlation function in either
    (μ, s) or (rₚ, π) coordinates and save the resulting map to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    x_array : np.ndarray
        Array containing the x-coordinate values (μ or rₚ) for each point.
    y_array : np.ndarray
        Array containing the y-coordinate values (s or π) for each point.
    z_array : np.ndarray
        Array of 2-points correlation function values arranged consistently with
        `(x_array, y_array)`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    kind : str
        Type of correlation map to plot. Must be either `"measured"` or
        `"correct"`. Used for labeling the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    v_min : float, optional
        Minimum value for the color scale.
    v_max : float, optional
        Maximum value for the color scale.
    lvls : int, optional
        Number of contour levels used in the filled contour plot.
    draw_lines : bool, optional
        If `True`, contour lines are drawn on top of the filled map.

    Returns
    -------
    tuple
        A tuple `(X, Y, Z)` where:
        - `X` and `Y` are the meshgrid arrays built from the unique values
          of the x and y coordinates.
        - `Z` is the reshaped correlation map corresponding to `(X, Y)`,
          transposed when `coords == "RpPI"`.
    """

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        SR = r'(\mu, s)'
        x_label = r'$\mu$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: s is SLOW (Y-axis), mu is FAST (X-axis)
        # Native shape (N_s, N_mu) == Plotting shape (N_y, N_x)  =>  no transpose needed
        # N.B.: y_unique corresponds to the SLOW index 's', x_unique to the FAST index 'mu'
        x_unique = np.unique(x_array) # N_mu
        y_unique = np.unique(y_array) # N_s
        Z = z_array.reshape((len(y_unique), len(x_unique)))

    elif coords == "RpPI":
        SR = r'(r_\mathrm{p}, \pi)'
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$\pi \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: r_p is SLOW, pi is FAST
        # Native shape (N_rp, N_pi) != Plotting shape (N_pi, N_rp)  =>  tranposition is required
        # N.B.: x_unique corresponds to the SLOW index 'r_p', y_unique to the FAST index 'pi'
        x_unique = np.unique(x_array) # N_rp
        y_unique = np.unique(y_array) # N_pi
        
        # reshape to native structure (r_p-rows, pi-columns)
        Z_native = z_array.reshape((len(x_unique), len(y_unique)))
        
        # transpose to plot structure (pi-rows, r_p-columns)
        Z = Z_native.T

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")

    X, Y = np.meshgrid(x_unique, y_unique)

    plt.figure(figsize=(9,8), num=f"2D map {coords} {kind}")
    contour = plt.contourf(X, Y, Z, levels=lvls, cmap='turbo', vmin=v_min, vmax=v_max)
    
    # contour lines
    if draw_lines:
        lines = plt.contour(X, Y, Z, levels=lvls, colors='black', linewidths=0.6)
        # plt.clabel(lines, inline=True, fontsize=8, fmt="%.2f", colors='white')

    cbar = plt.colorbar(contour, label=fr'$\xi^\mathrm{{{kind}}}{SR}$')
    # xi_ticks = np.linspace(np.min(xi_array), np.max(xi_array), 9)
    # cbar.set_ticks(xi_ticks)
    # cbar.set_ticklabels([f"{tick:.2f}" for tick in xi_ticks])

    # mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 5)
    # s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 6)
    # plt.xticks(mu_ticks, [f"{tick:.0f}" for tick in mu_ticks])
    # plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fr'2D map of $\xi^\mathrm{{{kind}}}{SR}$')
    plt.tight_layout()
    plt.savefig(f"{base_path}/2Dmap{coords}_{kind}.pdf", dpi=600)

    return X, Y, Z


def plot_contourf_ratio(coords: str,
                        x_array: npt.NDArray[np.float64],
                        y_array: npt.NDArray[np.float64],
                        z_array_ratio: npt.NDArray[np.float64],
                        base_path: str,
                        xlim: Optional[Tuple[float, float]] = None,
                        ylim: Optional[Tuple[float, float]] = None,
                        lvls: int = 20,
                        v_min: Optional[float] = None,
                        v_max: Optional[float] = None,
                        z_min: Optional[float] = None,
                        z_max: Optional[float] = None,
                        draw_lines: bool = False
) -> None:
    """
    Generate a filled contour plot of the ratio between measured and
    correct correlation functions in either (μ, s) or (rₚ, π) coordinates,
    and save the resulting map to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    x_array : np.ndarray
        Array containing the x-coordinate values (μ or rₚ) for each point.
    y_array : np.ndarray
        Array containing the y-coordinate values (s or π) for each point.
    z_array_ratio : np.ndarray
        Array of the ratio between measured and correct 2-points correlation
        function values arranged consistently with `(x_array, y_array)`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    lvls : int, optional
        Number of contour levels used in the filled contour plot.
    v_min : float, optional
        Minimum value for the color scale.
    v_max : float, optional
        Maximum value for the color scale.
    z_min : float, optional
        Values with absolute magnitude below this threshold are masked
        (set to NaN) in the ratio map.
    z_max : float, optional
        Values with absolute magnitude above this threshold are masked
        (set to NaN) in the ratio map.
    draw_lines : bool, optional
        If `True`, contour lines are drawn on top of the filled map.

    Returns
    -------
    None
        The function plots the contourf plot of the ratio between measured and correct
        correlation function.
    """
    
    x_unique = np.unique(x_array)
    y_unique = np.unique(y_array)

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        SR = r'(\mu, s)'
        x_label = r'$\mu$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: s is SLOW (Y-axis), mu is FAST (X-axis).
        # Native shape (N_s, N_mu) == Plotting shape (N_y, N_x)  =>  no transpose needed
        # Reshape directly to (N_y, N_x)
        Z = z_array_ratio.reshape((len(y_unique), len(x_unique)))

    elif coords == "RpPI":
        SR = r'(r_\mathrm{p}, \pi)'
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$\pi \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: r_p is SLOW, pi is FAST.
        # Native shape (N_rp, N_pi) != Plotting shape (N_pi, N_rp)  =>  tranposition is required
        
        # reshape to native structure (r_p-rows, pi-columns)
        Z_native = z_array_ratio.reshape((len(x_unique), len(y_unique)))
        
        # transpose to plot structure (pi-rows, r_p-columns)
        Z = Z_native.T

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    X, Y = np.meshgrid(x_unique, y_unique)

    # discard all values outside (z_min, z_max)
    if z_min is not None:
        mask = np.abs(Z) < z_min
        Z[mask] = np.nan
    
    if z_max is not None:
        mask = np.abs(Z) > z_max
        Z[mask] = np.nan

    plt.figure(figsize=(9,8), num=f"2D map {coords} ratio")
    contour = plt.contourf(X, Y, Z, levels=lvls, cmap='RdBu_r', vmin=v_min, vmax=v_max)

    # contour lines
    if draw_lines:
        lines = plt.contour(X, Y, Z, levels=lvls, colors='black', linewidths=0.6)
        # plt.clabel(lines, inline=True, fontsize=8, fmt="%.2f", colors='white')

    cbar = plt.colorbar(contour, label=fr'$\frac{{\xi_\mathrm{{measured}}}}{{\xi_\mathrm{{correct}}}} {SR}$')
    # xi_ticks = np.linspace(np.min(xi_array), np.max(xi_array), 9)
    # cbar.set_ticks(xi_ticks)
    # cbar.set_ticklabels([f"{tick:.2f}" for tick in xi_ticks])

    # mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 5)
    # s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 6)
    # plt.xticks(mu_ticks, [f"{tick:.0f}" for tick in mu_ticks])
    # plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fr'2D map of $\frac{{\xi_\mathrm{{measured}}}}{{\xi_\mathrm{{correct}}}} {SR}$')
    plt.tight_layout()
    plt.savefig(f"{base_path}/2Dmap{coords}_ratio.pdf", dpi=600)


def plot_imshow(coords: str,
                x_array: npt.NDArray[np.float64],
                y_array: npt.NDArray[np.float64],
                z_array: npt.NDArray[np.float64],
                base_path: str,
                kind: str,
                xlim: Optional[Tuple[float, float]] = None,
                ylim: Optional[Tuple[float, float]] = None,
                v_min: Optional[float] = None,
                v_max: Optional[float] = None,
                interp: str = "nearest"
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Generate an image plot (imshow) of the correlation function in either
    (μ, s) or (rₚ, π) coordinates, optionally setting color limits, and
    save the resulting figure to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    x_array : np.ndarray
        Array containing the x-coordinate values (μ or rₚ) for each point.
    y_array : np.ndarray
        Array containing the y-coordinate values (s or π) for each point.
    z_array : np.ndarray
        Array of 2-points correlation function values arranged consistently with
        `(x_array, y_array)`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    kind : str
        Type of correlation map to plot. Must be either `"measured"` or
        `"correct"`. Used for labeling the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    v_min : float, optional
        Minimum value for the color scale.
        from imshow plot.
    v_max : float, optional
        Maximum value for the color scale.
    interp : str, optional
        Interpolation formula used in imshow. Default is `"nearest"`.
        Typical formulae: "nearest" (no smoothing), "bilinear" (smooth gradient), "bicubic" (even smoother).

    Returns
    -------
    tuple
        A tuple `(X, Y, Z)` where:
        - `X` and `Y` are the meshgrid arrays built from the unique values
          of the x and y coordinates.
        - `Z` is the reshaped correlation map corresponding to `(X, Y)`,
          transposed when `coords == "RpPI"`.
    """

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        SR = r'(\mu, s)'
        x_label = r'$\mu$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: s is SLOW (Y-axis), mu is FAST (X-axis)
        # Native shape (N_s, N_mu) == Plotting shape (N_y, N_x)  =>  no transpose needed
        # N.B.: y_unique corresponds to the SLOW index 's', x_unique to the FAST index 'mu'
        x_unique = np.unique(x_array) # N_mu
        y_unique = np.unique(y_array) # N_s
        Z = z_array.reshape((len(y_unique), len(x_unique)))

    elif coords == "RpPI":
        SR = r'(r_\mathrm{p}, \pi)'
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$\pi \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: r_p is SLOW, pi is FAST
        # Native shape (N_rp, N_pi) != Plotting shape (N_pi, N_rp)  =>  tranposition is required
        # N.B.: x_unique corresponds to the SLOW index 'r_p', y_unique to the FAST index 'pi'
        x_unique = np.unique(x_array) # N_rp
        y_unique = np.unique(y_array) # N_pi
        
        # reshape to native structure (r_p-rows, pi-columns)
        Z_native = z_array.reshape((len(x_unique), len(y_unique)))
        
        # transpose to plot structure (pi-rows, r_p-columns)
        Z = Z_native.T

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")

    X, Y = np.meshgrid(x_unique, y_unique)

    # this is for the imshow to have physical coordinate axes instead of pixels
    extent = [
        x_unique.min(), x_unique.max(),
        y_unique.min(), y_unique.max()
    ]

    plt.figure(figsize=(9, 8), num=f"imshow {coords} {kind}")

    img = plt.imshow(Z,
                     origin="lower",
                     extent=extent, # type: ignore
                     cmap="turbo",
                     aspect="auto",
                     vmin=v_min,
                     vmax=v_max,
                     interpolation=interp)

    cbar = plt.colorbar(img, label=fr'$\xi^\mathrm{{{kind}}}{SR}$')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fr'imshow of $\xi^\mathrm{{{kind}}}{SR}$')
    plt.tight_layout()
    plt.savefig(f"{base_path}/imshow{coords}_{kind}.pdf", dpi=600)

    return X, Y, Z


def plot_imshow_ratio(coords: str,
                      x_array: npt.NDArray[np.float64],
                      y_array: npt.NDArray[np.float64],
                      z_array_ratio: npt.NDArray[np.float64],
                      base_path: str,
                      xlim: Optional[Tuple[float, float]] = None,
                      ylim: Optional[Tuple[float, float]] = None,
                      v_min: Optional[float] = None,
                      v_max: Optional[float] = None,
                      z_min: Optional[float] = None,
                      z_max: Optional[float] = None,
                      interp: str = "nearest"
) -> None:
    """
    Generate an image plot (imshow) of the ratio between measured and
    correct correlation functions in either (μ, s) or (rₚ, π) coordinates,
    optionally setting color limits and masking extreme values, and save the
    resulting figure to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    x_array : np.ndarray
        Array containing the x-coordinate values (μ or rₚ) for each point.
    y_array : np.ndarray
        Array containing the y-coordinate values (s or π) for each point.
    z_array_ratio : np.ndarray
        Array of the ratio between measured and correct 2-points correlation
        function values arranged consistently with `(x_array, y_array)`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    kind : str
        Type of correlation map to plot. Must be either `"measured"` or
        `"correct"`. Used for labeling the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    v_min : float, optional
        Minimum value for the color scale.
    v_max : float, optional
        Maximum value for the color scale.
    z_min : float, optional
        Values with absolute magnitude below this threshold are masked
        (set to NaN) in the ratio map.
    z_max : float, optional
        Values with absolute magnitude above this threshold are masked
        (set to NaN) in the ratio map.
    interp : str, optional
        Interpolation formula used in imshow. Default is `"nearest"`.
        Typical formulae: "nearest" (no smoothing), "bilinear" (smooth gradient), "bicubic" (even smoother).

    Returns
    -------
    None
        The function plots the contourf plot of the ratio between measured and correct
        correlation function.
    """

    x_unique = np.unique(x_array)
    y_unique = np.unique(y_array)

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        SR = r'(\mu, s)'
        x_label = r'$\mu$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: s is SLOW (Y-axis), mu is FAST (X-axis).
        # Native shape (N_s, N_mu) == Plotting shape (N_y, N_x)  =>  no transpose needed
        # Reshape directly to (N_y, N_x)
        Z = z_array_ratio.reshape((len(y_unique), len(x_unique)))

    elif coords == "RpPI":
        SR = r'(r_\mathrm{p}, \pi)'
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$\pi \,[h^{-1} \, \mathrm{Mpc}]$'
        
        # in this case: r_p is SLOW, pi is FAST.
        # Native shape (N_rp, N_pi) != Plotting shape (N_pi, N_rp)  =>  tranposition is required
        
        # reshape to native structure (r_p-rows, pi-columns)
        Z_native = z_array_ratio.reshape((len(x_unique), len(y_unique)))
        
        # transpose to plot structure (pi-rows, r_p-columns)
        Z = Z_native.T

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    # discard all values outside (z_min, z_max)
    if z_min is not None:
        mask = np.abs(Z) < z_min
        Z[mask] = np.nan
    
    if z_max is not None:
        mask = np.abs(Z) > z_max
        Z[mask] = np.nan
    
    # this is for the imshow to have physical coordinate axes instead of pixels
    extent = [
        x_unique.min(), x_unique.max(),
        y_unique.min(), y_unique.max()
    ]

    plt.figure(figsize=(9, 8), num=f"imshow {coords} ratio")

    img = plt.imshow(Z,
                     origin="lower",
                     extent=extent, # type: ignore
                     cmap="RdBu_r",
                     aspect="auto",
                     vmin=v_min,
                     vmax=v_max,
                     interpolation=interp)

    cbar = plt.colorbar(img, label=fr'$\frac{{\xi_\mathrm{{measured}}}}{{\xi_\mathrm{{correct}}}} {SR}$')
    
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fr'imshow of $\frac{{\xi_\mathrm{{measured}}}}{{\xi_\mathrm{{correct}}}} {SR}$')
    plt.tight_layout()
    plt.savefig(f"{base_path}/imshow{coords}_ratio.pdf", dpi=600)


def plot_covarianceMatrix(coords: str,
                          scale: npt.NDArray[np.float64],
                          cov: npt.NDArray[np.float64],
                          base_path: str,
                          kind: str,
                          fig: str,
                          filename: str,
                          l_values: Optional[npt.NDArray[np.int8]] = None,
                          n_wedges: Optional[int] = None,
                          xlim: Optional[Tuple[float, float]] = None,
                          ylim: Optional[Tuple[float, float]] = None,
                          v_min: Optional[float] = None,
                          v_max: Optional[float] = None,
                          interp: str = "nearest"
) -> None:
    """
    Generate an image plot (imshow) of the given covariance matrix,
    optionally setting color limits, and save the resulting figure to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    scale : np.ndarray
        Array containing the unique scale values (s or rₚ) for each point.
    cov : np.ndarray
        Covariance matrix, shaped like `(len(scale), len(scale))`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    kind : str
        Type of correlation map to plot. Must be either `"measured"`,
        `"correct"`. Used for labeling the plot.
    fig : str
        `num` value to feed `plt.figure`.
    filename : str
        Name for the file to be saved.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    v_min : float, optional
        Minimum value for the color scale
        from imshow plot.
    v_max : float, optional
        Maximum value for the color scale.
    interp : str, optional
        Interpolation formula used in imshow. Default is `"nearest"`.
        Typical formulae: "nearest" (no smoothing), "bilinear" (smooth gradient), "bicubic" (even smoother).

    Returns
    -------
    None
        The functon plots the covariance matrix.
    """

    if l_values is not None and n_wedges is not None:
        raise ValueError("Please provide either 'l_values' OR 'n_wedges', not both.")

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        x_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'

    elif coords == "RpPI":
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")

    # finding where the scale resets (or decreases), indicating a new block
    # diff < 0 means the scale went from high -> low (start of new multipole/wedge)
    n_bins = len(scale)
    boundaries = np.where(np.diff(scale) < 0)[0] + 1
    boundaries = np.concatenate(([0], boundaries, [n_bins]))
    n_blocks = len(boundaries) - 1

    # check for mismatches
    if n_wedges is not None and n_blocks != n_wedges:
        raise ValueError(f"Warning: n_wedges={n_wedges} passed, but detected {n_blocks} blocks in the data.")
    if l_values is not None and n_blocks != len(l_values):
        raise ValueError(f"Warning: {len(l_values)} l_values passed, but detected {n_blocks} blocks in the data.")

    fig_obj, ax = plt.subplots(figsize=(9, 8), num=fig + " " + kind)

    img = plt.imshow(cov,
                     origin="lower",
                     cmap="RdBu_r",
                     aspect="equal",
                     vmin=v_min,
                     vmax=v_max,
                     interpolation=interp)

    cbar = plt.colorbar(img, label=fr'$C^\mathrm{{{kind}}}$')

    first_block_scale = scale[boundaries[0]:boundaries[1]]    
    desired_ticks = np.linspace(first_block_scale.min(), first_block_scale.max(), 6)[1:-1]

    # find the matrix indices corresponding to these physical values
    tick_indices = []
    tick_labels = []
    block_centers = []
    block_labels = []

    # iterate over each multipole block to place ticks
    for i in range(n_blocks):
        start = boundaries[i]
        end = boundaries[i+1]
        
        # get scale for this block
        block_scale = scale[start:end]
        
        # interpolate to find pixel index for each desired physical value
        # indices are 0..(end-start), mapped to block_scale values
        idxs = np.interp(desired_ticks, block_scale, np.arange(len(block_scale)))
        
        # add the offset of the block start
        tick_indices.extend(idxs + start)
        
        # add ticks
        tick_labels.extend([f"{t:.1f}" for t in desired_ticks])
        
        # draw separator lines
        if end < n_bins:
            plt.axhline(end - 0.5, color='black', linestyle='-', linewidth=1)
            plt.axvline(end - 0.5, color='black', linestyle='-', linewidth=1)

        block_centers.append((start + end - 1) / 2.0)
        if l_values is not None:
            block_labels.append(fr"$\ell={l_values[i]}$")

        elif n_wedges is not None:
            mu_step = 1.0 / n_wedges
            mu_low = i * mu_step
            mu_high = (i + 1) * mu_step
            block_labels.append(fr"$[{mu_low:.2f}, {mu_high:.2f}]$")

    # primary axes (scale)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # secondary axes (multipole/wedge labels)
    # we use secondary axes to place labels at the top and right
    ax_top = ax.secondary_xaxis('top')
    ax_right = ax.secondary_yaxis('right')
    
    ax_top.set_xticks(block_centers)
    ax_top.set_xticklabels(block_labels, fontsize=12, fontweight='bold')
    ax_right.set_yticks(block_centers)
    ax_right.set_yticklabels(block_labels, fontsize=12, fontweight='bold', rotation=270, va='center')

    # clean up secondary ticks (hide the little tick marks, keep labels)
    ax_top.tick_params(axis='x', which='both', length=0)
    ax_right.tick_params(axis='y', which='both', length=0)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(fig + " " + kind)
    plt.tight_layout()
    plt.savefig(f"{base_path}/{filename}.pdf", dpi=600)


def plot_correlationMatrix(coords: str,
                           scale: npt.NDArray[np.float64],
                           cov: npt.NDArray[np.float64],
                           base_path: str,
                           kind: str,
                           fig: str,
                           filename: str,
                           l_values: Optional[npt.NDArray[np.int8]] = None,
                           n_wedges: Optional[int] = None,
                           xlim: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None,
                           v_min: float = -1.0,
                           v_max: float = 1.0,
                           interp: str = "nearest"
) -> None:
    """
    Generate an image plot (imshow) of the correlation matrix derived from 
    the given covariance matrix, and save the resulting figure to file.

    Parameters
    ----------
    coords : str
        Coordinate system of the input data. Must be either `"SMU"` for
        (μ, s) or `"RpPI"` for (rₚ, π).
    scale : np.ndarray
        Array containing the unique scale values (s or rₚ) for each point.
    cov : np.ndarray
        Covariance matrix, shaped like `(len(scale), len(scale))`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    kind : str
        Type of correlation map to plot. Must be either `"measured"`,
        `"correct"`. Used for labeling the plot.
    fig : str
        `num` value to feed `plt.figure`.
    filename : str
        Name for the file to be saved.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.
    v_min : float, optional
        Minimum value for the color scale from imshow plot.
        Default is `-1.0`, determined by the definition of the correlation matrix.
    v_max : float, optional
        Maximum value for the color scale.
        Default is `1.0`, determined by the definition of the correlation matrix.
    interp : str, optional
        Interpolation formula used in imshow. Default is `"nearest"`.
        Typical formulae: "nearest" (no smoothing), "bilinear" (smooth gradient), "bicubic" (even smoother).

    Returns
    -------
    None
        The functon computes and plots the correlation matrix.
    """

    if l_values is not None and n_wedges is not None:
        raise ValueError("Please provide either 'l_values' OR 'n_wedges', not both.")

    # covariance to correlation matrix
    std = np.sqrt(np.diag(cov))
    outer_v = np.outer(std, std)
    corr = np.divide(cov, outer_v, out=np.zeros_like(cov), where=outer_v!=0)

    # check for coords to be either "SMU" or "RpPI"
    valid_coords = ["SMU", "RpPI"]
    if coords == "SMU":
        x_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$s \,[h^{-1} \, \mathrm{Mpc}]$'

    elif coords == "RpPI":
        x_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'
        y_label = r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$'

    else:
        raise ValueError(f"coords must be one of {valid_coords}, got '{coords}'")

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")
    
    # finding where the scale resets (or decreases), indicating a new block
    # diff < 0 means the scale went from high -> low (start of new multipole/wedge)
    n_bins = len(scale)
    boundaries = np.where(np.diff(scale) < 0)[0] + 1
    boundaries = np.concatenate(([0], boundaries, [n_bins]))
    n_blocks = len(boundaries) - 1

    # check for mismatches
    if n_wedges is not None and n_blocks != n_wedges:
        raise ValueError(f"Warning: n_wedges={n_wedges} passed, but detected {n_blocks} blocks in the data.")
    if l_values is not None and n_blocks != len(l_values):
        raise ValueError(f"Warning: {len(l_values)} l_values passed, but detected {n_blocks} blocks in the data.")

    fig_obj, ax = plt.subplots(figsize=(9, 8), num=fig + " " + kind)

    img = plt.imshow(corr,
                     origin="lower",
                     cmap="RdBu_r",
                     aspect="equal",
                     vmin=v_min,
                     vmax=v_max,
                     interpolation=interp)

    cbar = plt.colorbar(img, label=fr'$R^\mathrm{{{kind}}}$', ax= ax)

    first_block_scale = scale[boundaries[0]:boundaries[1]]    
    desired_ticks = np.linspace(first_block_scale.min(), first_block_scale.max(), 6)[1:-1]

    # find the matrix indices corresponding to these physical values
    tick_indices = []
    tick_labels = []
    block_centers = []
    block_labels = []

    # iterate over each multipole block to place ticks
    for i in range(n_blocks):
        start = boundaries[i]
        end = boundaries[i+1]
        
        # get scale for this block
        block_scale = scale[start:end]
        
        # interpolate to find pixel index for each desired physical value
        # indices are 0..(end-start), mapped to block_scale values
        idxs = np.interp(desired_ticks, block_scale, np.arange(len(block_scale)))
        
        # add the offset of the block start
        tick_indices.extend(idxs + start)
        
        # add ticks
        tick_labels.extend([f"{t:.1f}" for t in desired_ticks])
        
        # draw separator lines
        if end < n_bins:
            plt.axhline(end - 0.5, color='black', linestyle='-', linewidth=1)
            plt.axvline(end - 0.5, color='black', linestyle='-', linewidth=1)

        block_centers.append((start + end - 1) / 2.0)
        if l_values is not None:
            block_labels.append(fr"$\ell={l_values[i]}$")

        elif n_wedges is not None:
            mu_step = 1.0 / n_wedges
            mu_low = i * mu_step
            mu_high = (i + 1) * mu_step
            block_labels.append(fr"$[{mu_low:.2f}, {mu_high:.2f}]$")

    # primary axes (scale)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # secondary axes (multipole/wedge labels)
    # we use secondary axes to place labels at the top and right
    ax_top = ax.secondary_xaxis('top')
    ax_right = ax.secondary_yaxis('right')
    
    ax_top.set_xticks(block_centers)
    ax_top.set_xticklabels(block_labels, fontsize=12, fontweight='bold')
    ax_right.set_yticks(block_centers)
    ax_right.set_yticklabels(block_labels, fontsize=12, fontweight='bold', rotation=270, va='center')

    # clean up secondary ticks (hide the little tick marks, keep labels)
    ax_top.tick_params(axis='x', which='both', length=0)
    ax_right.tick_params(axis='y', which='both', length=0)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(fig + " " + kind)
    plt.tight_layout()
    plt.savefig(f"{base_path}/{filename}.pdf", dpi=600)