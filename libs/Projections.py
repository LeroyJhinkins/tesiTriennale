import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks
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


# ================================================================================================================================= #
#                                                                                                                                   #
#                                                        Multipoles projection                                                      #
#                                                                                                                                   #
# ================================================================================================================================= #
# source: https://arxiv.org/pdf/1205.5573
def legendre(l, mu):
    
    coeffs = np.zeros(l + 1)
    coeffs[l] = 1.0
    return np.polynomial.legendre.Legendre(coeffs)(mu)


# XI is a function of s and mu (:= cosθ)
# so in this dataset we have several values of XI(s, mu) -> for every value of s we have 200 values of XI, for -1 < mu < 1
# therefore we extract mu and XI values for every s
def compute_multipoles(l_values: npt.NDArray[np.float64],
                       s_array: npt.NDArray[np.float64],
                       mu_array: npt.NDArray[np.float64],
                       xi_array: npt.NDArray[np.float64],
                       delta_mu: float = 0.01
) -> npt.NDArray[np.float64]:
    """
    Compute the Legendre multipoles of the correlation function ξ(s, μ)
    by projecting it onto the specified set of Legendre polynomials.

    Parameters
    ----------
    l_values : np.ndarray
        Array of multipole orders ℓ for which to compute the Legendre
        coefficients.
    s_array : np.ndarray
        Array containing the separation values s for each data point.
    mu_array : np.ndarray
        Array of μ values associated with the same elements of `s_array`.
    xi_array : np.ndarray
        Array of 2-points correlation function values ξ(s, μ) corresponding to each
        pair (s, μ).

    Returns
    -------
    np.ndarray
        A 2D array of shape `(len(s_unique), len(l_values))`, where each
        row corresponds to a unique separation s and each column to a
        multipole order ℓ. The entry `[i, j]` is the j-th multipole of
        ξ at the i-th separation value.
    """
    
    s_unique = np.unique(s_array)
    xi_multipoles = np.zeros((len(s_unique), len(l_values)))

    for i, s in enumerate(s_unique):
            
        mask = (s_array == s)
        mu_vals = mu_array[mask]
        xi_vals = xi_array[mask]

        for j, l in enumerate(l_values):
                
            P_l_array = legendre(l, mu_vals)
            xi_multipoles[i,j] = ((2*l + 1)/2 * np.sum(xi_vals * P_l_array) * delta_mu) # l-th coefficient of the Legendre projection
                                                                                        # for the i-th value of XI

    return xi_multipoles


def print_multipoles(l_values: npt.NDArray[np.float64],
                     s_array: npt.NDArray[np.float64],
                     xi_multipoles: npt.NDArray[np.float64],
                     n_values: int,
                     prec: int = 10
) -> None:
    """
    Print the first `n_values` separation points and their corresponding
    multipole values of the correlation function ξ(s, μ).

    Parameters
    ----------
    l_values : np.ndarray
        Array of multipole orders ℓ corresponding to the columns of
        `xi_multipoles`.
    s_array : np.ndarray
        Array containing the separation values s for each data point.
    xi_multipoles : np.ndarray
        Array of computed multipoles, as returned by `compute_multipoles`.
        Shape should be `(len(unique(s_array)), len(l_values))`.
    n_values : int
        Number of separation points to print.
    prec : int, optional
        Number of decimal places (in scientific notation) for printing
        the multipole values. Default is 10.

    Returns
    -------
    None
        The function prints the values to standard output.
    """

    xi_strings = [f"XI{l}" for l in l_values]
    print("  s " + " ".join(xi_strings)) # e.g: this will print "s XI0 XI1 ..."
    
    s_unique = np.unique(s_array)

    for s, xi in zip(s_unique[:n_values], xi_multipoles[:n_values, :]):
        xi_str = " ".join(f"{x:.{prec}e}" for x in xi)
        print(f"  {s:.1f} {xi_str}")


def plot_multipole(l_value: int,
                   s_unique: npt.NDArray[np.float64],
                   multipole: npt.NDArray[np.float64],
                   base_path: str,
                   err_multipole: Optional[npt.NDArray[np.float64]] = None,
                   xlim: Optional[Tuple[float, float]] = None,
                   ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot a single multipole of the correlation function as a function of
    separation s, optionally including error bars, and save the figure to file.

    Parameters
    ----------
    l_value : int
        Multipole order ℓ corresponding to the `multipole` values.
    s_unique : np.ndarray
        Array of unique separation values s at which the multipole is evaluated.
    multipole : np.ndarray
        Array containing the multipole values for each separation in `s_unique`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_multipole : np.ndarray, optional
        Error estimates associated with the multipole values. If provided,
        error bars are included in the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot named `multipole{l_value}.pdf`
        in the specified directory.
    """

    plt.figure(figsize=(8, 8), num="Multipole plot")

    label = fr"$\xi_{l_value}$"

    y = (s_unique**2) * multipole
    
    if err_multipole is not None:
            err_y = (s_unique**2) * err_multipole
    else:
        err_y = None

    plt.errorbar(s_unique, y, yerr=err_y, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(fr"Multipole $l={l_value}$")
    
    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(fr'$s^2 \xi_{l_value} \,[h^{{-2}} \, \mathrm{{Mpc}}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/multipole{l_value}.pdf", dpi=600)


def plot_multipole_measVScorr(l_value: int,
                              s_unique: npt.NDArray[np.float64],
                              multipole_measured: npt.NDArray[np.float64],
                              multipole_correct: npt.NDArray[np.float64],
                              base_path: str,
                              err_multipole_measured: Optional[npt.NDArray[np.float64]] = None,
                              err_multipole_correct: Optional[npt.NDArray[np.float64]] = None,
                              xlim: Optional[Tuple[float, float]] = None,
                              ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot and compare the measured and corrected multipoles of the correlation
    function as a function of separation s, optionally including error bars,
    and save the figure to file.

    Parameters
    ----------
    l_value : int
        Multipole order ℓ corresponding to the multipole arrays.
    s_unique : np.ndarray
        Array of unique separation values s at which the multipoles are evaluated.
    multipole_measured : np.ndarray
        Array containing the measured multipole values for each separation in `s_unique`.
    multipole_correct : np.ndarray
        Array containing the corrected multipole values for each separation in `s_unique`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_multipole_measured : np.ndarray, optional
        Error estimates associated with the measured multipoles. If provided,
        error bars are included in the plot.
    err_multipole_correct : np.ndarray, optional
        Error estimates associated with the corrected multipoles. If provided,
        error bars are included in the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot named
        `multipole{l_value}_measVScorr.pdf` in the specified directory.
    """

    plt.figure(figsize=(8, 8), num="Multipole plot Measured VS Correct")

    label_measured = fr"$\xi_{l_value}^{{\mathrm{{measured}}}}$"
    label_correct = fr"$\xi_{l_value}^{{\mathrm{{correct}}}}$"

    y_measured = (s_unique**2) * multipole_measured
    y_correct = (s_unique**2) * multipole_correct

    if err_multipole_measured is not None:
        err_y_measured = (s_unique**2) * err_multipole_measured
    else:
        err_y_measured = None

    if err_multipole_correct is not None:
        err_y_correct = (s_unique**2) * err_multipole_correct
    else:
        err_y_correct = None

    plt.errorbar(s_unique, y_measured, yerr=err_y_measured, label=label_measured, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)
    plt.errorbar(s_unique, y_correct, yerr=err_y_correct, label=label_correct, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(fr"Multipole $l={l_value}$ Measured VS Correct")
    
    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(fr'$s^2 \xi_{l_value} \,[h^{{-2}} \, \mathrm{{Mpc}}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/multipole{l_value}_measVScorr.pdf", dpi=600)


def plot_multipole_ratio(l_value: int,
                         s_unique: npt.NDArray[np.float64],
                         multipole_measured: npt.NDArray[np.float64],
                         multipole_correct: npt.NDArray[np.float64],
                         base_path: str,
                         err_multipole_measured: Optional[npt.NDArray[np.float64]] = None,
                         err_multipole_correct: Optional[npt.NDArray[np.float64]] = None,
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None
) -> Tuple[
    npt.NDArray[np.float64],
    Optional[npt.NDArray[np.float64]]
]:
    """
    Compute and plot the ratio of measured to corrected multipoles of the correlation
    function as a function of separation s, optionally including propagated
    errors, and save the figure to file.

    Parameters
    ----------
    l_value : int
        Multipole order ℓ corresponding to the multipoles.
    s_unique : np.ndarray
        Array of unique separation values s at which the multipoles are evaluated.
    multipole_measured : np.ndarray
        Array of measured multipole values for each separation in `s_unique`.
    multipole_correct : np.ndarray
        Array of corrected multipole values for each separation in `s_unique`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_multipole_measured : np.ndarray, optional
        Error estimates associated with the measured multipoles. If provided,
        they are used to compute the propagated error of the ratio.
    err_multipole_correct : np.ndarray, optional
        Error estimates associated with the corrected multipoles. If provided,
        they are used to compute the propagated error of the ratio.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    tuple
        A tuple `(multipole_ratio, err_ratio)` where:
        - `multipole_ratio` is an array of the element-wise ratio
          `multipole_measured / multipole_correct`.
        - `err_ratio` is an array of the propagated errors for the ratio, or
          `None` if error arrays were not provided.
    """

    multipole_ratio = np.divide(
        multipole_measured,
        multipole_correct,
        out=np.full_like(multipole_measured, np.nan),
        where=(multipole_correct != 0)
    )

    if err_multipole_measured is not None and err_multipole_correct is not None:
        
        term1 = np.divide(
            err_multipole_measured, multipole_correct,
            out=np.zeros_like(err_multipole_measured),
            where=(multipole_correct != 0)
        )

        term2 = np.divide(
            multipole_measured * err_multipole_correct, multipole_correct**2,
            out=np.zeros_like(err_multipole_measured),
            where=(multipole_correct != 0)
        )

        err_ratio = np.sqrt(term1**2 + term2**2)

    else:
        err_ratio = None

    plt.figure(figsize=(8, 8), num="Ratio multipole plot")

    label = fr"$\frac{{\xi_{l_value}^{{\mathrm{{measured}}}}}}{{\xi_{l_value}^{{\mathrm{{correct}}}}}}$"

    plt.errorbar(s_unique, multipole_ratio, yerr=err_ratio, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(fr"Ratio multipole $l={l_value}$")
    
    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(fr'$\frac{{\xi_{l_value}^{{\mathrm{{measured}}}}}}{{\xi_{l_value}^{{\mathrm{{correct}}}}}}$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/multipole{l_value}_ratio.pdf", dpi=600)

    return multipole_ratio, err_ratio


# ================================================================================================================================= #
#                                                                                                                                   #
#                                                        Clustering wedges                                                          #
#                                                                                                                                   #
# ================================================================================================================================= #
def compute_clusteringWedges(n_wedges: int,
                             s_array: npt.NDArray[np.float64],
                             mu_array: npt.NDArray[np.float64],
                             xi_array: npt.NDArray[np.float64],
                             delta_mu: float = 0.01
) -> npt.NDArray[np.float64]:
    """
    Compute clustering wedges by integrating the correlation function ξ(μ, s)
    over uniformly spaced μ-intervals.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_array : np.ndarray
        Array containing the separation values s for each data point.
    mu_array : np.ndarray
        Array of μ values associated with the same elements of `s_array`.
    xi_array : np.ndarray
        Array of 2-point correlation function values ξ(μ, s) corresponding to each
        pair (s, μ).

    Returns
    -------
    np.ndarray
        A 2D array of shape `(n_wedges, len(unique(s_array)))`. Each row
        contains the wedge associated with a specific μ-interval, and each
        column corresponds to a unique separation value s.
    """

    s_unique = np.unique(s_array)
    wedges = np.zeros((n_wedges, len(s_unique))) # so that the i-th line contains the i-th wedge
    
    mu_edges = np.linspace(0.0, 1.0, n_wedges + 1) # dividing the interval [0,1] in n_wedges intervals
                                                   # and the j-th column stands for the j-th value of s

    for j, s in enumerate(s_unique):

        mask = (s_array == s) & (mu_array >= 0)
        mu_vals = mu_array[mask]
        xi_vals = xi_array[mask]

        for w in range(n_wedges):

            mu_min = mu_edges[w]
            mu_max = mu_edges[w + 1]

            if w < n_wedges - 1:
                mask_wedge = (mu_vals >= mu_min) & (mu_vals < mu_max)
            else:
                mask_wedge = (mu_vals >= mu_min) & (mu_vals <= mu_max) # not to exclude 1 from the last interval

            wedges[w, j] = np.sum(xi_vals[mask_wedge]) * delta_mu * n_wedges # normalization is n_wedges because Δμ_bin = 1/n_wedges

    return wedges


def plot_clusteringWedges(n_wedges: int,
                          s_unique: npt.NDArray[np.float64],
                          wedges: npt.NDArray[np.float64],
                          base_path: str,
                          err_wedges: Optional[npt.NDArray[np.float64]] = None,
                          xlim: Optional[Tuple[float, float]] = None,
                          ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot clustering wedges as functions of the separation scale s and save
    the resulting figure to file.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_unique : np.ndarray
        Array of unique separation values s at which the wedges are evaluated.
    wedges : np.ndarray
        Array of shape `(n_wedges, len(s_unique))` containing the wedge
        values for each μ-bin.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_wedges : np.ndarray, optional
        Error estimates associated with each wedge, with the same shape as
        `wedges`. If provided, error bars are included in the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot named
        `"{n_wedges}clustWedges.pdf"` in the specified directory.
    """

    mu_edges = np.linspace(0.0, 1.0, n_wedges + 1)

    plt.figure(figsize=(8, 8), num="Clustering wedges")

    for w in range(n_wedges):

        mu_min = mu_edges[w]
        mu_max = mu_edges[w + 1]
        label = fr"$\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}$"

        y = (s_unique**2) * wedges[w]
        if err_wedges is not None:
            err_y = (s_unique**2) * err_wedges[w]
        else:
            err_y = None

        plt.errorbar(s_unique, y, yerr=err_y, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(f"{n_wedges} clustering wedges")
    
    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$s^2 \xi \,[h^{-2} \, \mathrm{Mpc}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/{n_wedges}clustWedges.pdf", dpi=600)


def plot_clusteringWedges_measVScorr(n_wedges: int,
                                     s_unique: npt.NDArray[np.float64],
                                     wedges_measured: npt.NDArray[np.float64],
                                     wedges_correct: npt.NDArray[np.float64],
                                     base_path: str,
                                     err_wedges_measured: Optional[npt.NDArray[np.float64]] = None,
                                     err_wedges_correct: Optional[npt.NDArray[np.float64]] = None,
                                     xlim: Optional[Tuple[float, float]] = None,
                                     ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot the measured clustering wedges together with the corresponding
    corrected wedges, and save the comparison figure to file.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_unique : np.ndarray
        Array of unique separation values s at which the wedges are evaluated.
    wedges_measured : np.ndarray
        Measured clustering wedges, with shape `(n_wedges, len(s_unique))`.
    wedges_correct : np.ndarray
        Corrected clustering wedges, shaped like `wedges_measured`.
    base_path : str
        Directory in which the output figure will be saved.
    err_wedges_measured : np.ndarray, optional
        Error estimates for the measured wedges. If provided, error bars
        are added to the measured curves.
    err_wedges_correct : np.ndarray, optional
        Error estimates for the corrected wedges, used to add error bars.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot comparing measured and
        corrected wedges as `"{n_wedges}clustWedges_measVScorr.pdf"` in
        the specified directory.
    """

    mu_edges = np.linspace(0.0, 1.0, n_wedges + 1)

    plt.figure(figsize=(8, 8), num="Clustering wedges Measured VS Correct")
    
    for w in range(n_wedges):

        mu_min = mu_edges[w]
        mu_max = mu_edges[w + 1]

        label_measured = fr"$\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}^\mathrm{{measured}}$"
        label_correct  = fr"$\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}^\mathrm{{correct}}$"

        y_measured = (s_unique**2) * wedges_measured[w]
        y_correct = (s_unique**2) * wedges_correct[w]

        if err_wedges_measured is not None:
            err_y_measured = (s_unique**2) * err_wedges_measured[w]
        else:
            err_y_measured = None

        if err_wedges_correct is not None:
            err_y_correct = (s_unique**2) * err_wedges_correct[w]
        else:
            err_y_correct = None

        plt.errorbar(s_unique, y_measured, yerr=err_y_measured, label=label_measured, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)
        plt.errorbar(s_unique, y_correct, yerr=err_y_correct, label=label_correct, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(f"{n_wedges} clustering wedges Measured VS Correct")
    
    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$s^2 \xi \,[h^{-2} \, \mathrm{Mpc}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/{n_wedges}clustWedges_measVScorr.pdf", dpi=600)


def plot_clusteringWedges_ratio(n_wedges: int,
                                s_unique: npt.NDArray[np.float64],
                                wedges_measured: npt.NDArray[np.float64],
                                wedges_correct: npt.NDArray[np.float64],
                                base_path: str,
                                err_wedges_measured: Optional[npt.NDArray[np.float64]] = None,
                                err_wedges_correct: Optional[npt.NDArray[np.float64]] = None,
                                xlim: Optional[Tuple[float, float]] = None,
                                ylim: Optional[Tuple[float, float]] = None
) -> Tuple[
    npt.NDArray[np.float64],
    Optional[npt.NDArray[np.float64]]
]:
    """
    Plot the ratio between measured and correct clustering wedges, compute
    the propagated uncertainties, and save the resulting figure to file.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_unique : np.ndarray
        Array of unique separation values s at which the wedges are evaluated.
    wedges_measured : np.ndarray
        Measured clustering wedges, with shape `(n_wedges, len(s_unique))`.
    wedges_correct : np.ndarray
        Corrected clustering wedges, shaped like `wedges_measured`.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_wedges_measured : np.ndarray, optional
        Error estimates for the measured wedges. If provided together with
        `err_wedges_correct`, error bars on the ratio are computed.
    err_wedges_correct : np.ndarray, optional
        Error estimates for the corrected wedges, used in the uncertainty
        propagation of the ratio.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    tuple
        A pair `(wedges_ratio, err_ratio)`, where:
        - `wedges_ratio` is the element-wise ratio between measured and
          corrected wedges, shaped `(n_wedges, len(s_unique))`.
        - `err_ratio` contains the propagated uncertainties of the ratio, or
          `None` if the error inputs are not provided.
    """

    mu_edges = np.linspace(0.0, 1.0, n_wedges + 1)

    wedges_ratio = np.divide(
        wedges_measured,
        wedges_correct,
        out=np.full_like(wedges_measured, np.nan),
        where=(wedges_correct != 0)
    )

    if err_wedges_measured is not None and err_wedges_correct is not None:
        
        term1 = np.divide(
            err_wedges_measured, wedges_correct,
            out=np.zeros_like(err_wedges_measured),
            where=(wedges_correct != 0)
        )

        term2 = np.divide(
            wedges_measured * err_wedges_correct, wedges_correct**2,
            out=np.zeros_like(err_wedges_measured),
            where=(wedges_correct != 0)
        )

        err_ratio = np.sqrt(term1**2 + term2**2)

    else:
        err_ratio = None

    plt.figure(figsize=(8, 8), num="Ratio Measured VS Correct")

    for w in range(n_wedges):

        mu_min = mu_edges[w]
        mu_max = mu_edges[w + 1]

        label = fr"$\frac{{\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}^\mathrm{{measured}}}}{{\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}^\mathrm{{correct}}}}$"

        y = wedges_ratio[w]
        
        if err_ratio is not None:
            err_y = err_ratio[w]
        else:
            err_y = None

        plt.errorbar(s_unique, y, yerr=err_y, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)    

    plt.title(f"{n_wedges} clustering wedges ratio")

    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$\frac{\xi^\mathrm{measured}}{\xi^\mathrm{correct}}$')
    
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim) # this is necessary because when wedge_correct is near 0, the ratio explodes => so we limit the plot on y-axis

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{base_path}/{n_wedges}clustWedge_ratio.pdf", dpi=600)

    return wedges_ratio, err_ratio


def compute_clusteringWedges_BAOpeaks(n_wedges: int,
                                      s_unique: npt.NDArray[np.float64],
                                      wedges: npt.NDArray[np.float64],
                                      err_wedges: Optional[npt.NDArray[np.float64]] = None,
                                      s_min: float = 50,
                                      s_max: float = 150
) -> Union[
    Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]],
    Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]]
]:
    """
    Identify the BAO peak position within a specified separation range for
    each clustering wedge, optionally computing confidence intervals when
    uncertainties are provided.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_unique : np.ndarray
        Array of unique separation values s at which the wedges are evaluated.
    wedges : np.ndarray
        Array of shape `(n_wedges, len(s_unique))` containing the wedge
        values for each μ-bin.
    err_wedges : np.ndarray, optional
        Error estimates associated with each wedge, with the same shape as
        `wedges`. If provided, confidence intervals around the BAO peak
        positions are computed.
    s_min : float, optional
        Lower bound of the separation range in which the BAO peak is searched.
    s_max : float, optional
        Upper bound of the separation range in which the BAO peak is searched.

    Returns
    -------
    tuple
        If no error estimates are provided:
            `(s_peak, xi_peak)`
        where:
            - `s_peak` contains the BAO peak position for each wedge.
            - `xi_peak` contains the corresponding peak values.

        If errors are provided:
            `(s_peak, xi_peak, s_low, s_high)`
        where:
            - `s_low` and `s_high` define the confidence interval bounds for
              each peak, obtained from the region where the wedge remains
              within 1σ of its maximum.
    """

    # restrict BAO window
    mask = (s_unique >= s_min) & (s_unique <= s_max)
    s_BAO = s_unique[mask]

    s_peak = np.zeros(n_wedges)
    xi_peak = np.zeros(n_wedges)

    # return peaks if no errors are provided
    if err_wedges is None:
        for w in range(n_wedges):
            xi_BAO = wedges[w, mask]

            # find local maxima
            peaks, _ = find_peaks(xi_BAO)

            if len(peaks) == 0:
                # fallback: global maximum inside the window
                idx = np.argmax(xi_BAO)
            else:
                # choose the highest local peak
                idx = peaks[np.argmax(xi_BAO[peaks])]

            s_peak[w] = s_BAO[idx]
            xi_peak[w] = xi_BAO[idx]

        return s_peak, xi_peak

    # if errors are provided
    s_low  = np.zeros(n_wedges)
    s_high = np.zeros(n_wedges)

    for w in range(n_wedges):
        xi_BAO = wedges[w, mask]
        err_BAO = err_wedges[w, mask]

        peaks, _ = find_peaks(xi_BAO)

        if len(peaks) == 0:
            idx = np.argmax(xi_BAO)
        else:
            idx = peaks[np.argmax(xi_BAO[peaks])]

        s_peak[w] = s_BAO[idx]
        xi_peak[w] = xi_BAO[idx]

        # confidence interval = all s where xi stays within 1σ of the peak value
        threshold = xi_peak[w] - err_BAO[idx]

        mask_conf = xi_BAO >= threshold
        s_conf = s_BAO[mask_conf]

        if s_conf.size == 0:
            s_low[w] = np.nan
            s_high[w] = np.nan
        else:
            s_low[w] = s_conf.min()
            s_high[w] = s_conf.max()

    return s_peak, xi_peak, s_low, s_high


def print_clusteringWedges_BAOintervals(n_wedges: int,
                                        s_peak: npt.NDArray[np.float64],
                                        xi_peak: npt.NDArray[np.float64],
                                        s_low: Optional[npt.NDArray[np.float64]] = None,
                                        s_high: Optional[npt.NDArray[np.float64]] = None
) -> None:
    """
    Print the BAO peak positions for each clustering wedge, optionally
    including the corresponding confidence intervals.

    Parameters
    ----------
    n_wedges : int
        Number of wedges to compute. The interval [0, 1] is divided into
        `n_wedges` bins of equal width.
    s_peak : np.ndarray
        BAO peak positions for each wedge, as returned by `compute_BAO_peaks`.
    xi_peak : np.ndarray
        Peak values of the wedges at the BAO positions.
    s_low : np.ndarray, optional
        Lower bounds of the confidence interval for each wedge, provided when
        uncertainties were included in the BAO peak computation.
    s_high : np.ndarray, optional
        Upper bounds of the confidence interval for each wedge.

    Returns
    -------
    None
        The function prints the peak positions, and if available, the
        confidence intervals associated with each clustering wedge.
    """

    mu_edges = np.linspace(0.0, 1.0, n_wedges + 1)

    for w in range(n_wedges):

        mu_min = mu_edges[w]
        mu_max = mu_edges[w + 1]

        print(f"  Wedge [{mu_min:.2f}, {mu_max:.2f}]:")
        print(f"    s_peak = {s_peak[w]:.3f}", end="")

        if s_low is not None and s_high is not None:
            print(f", s confidence interval: [{s_low[w]:.3f}, {s_high[w]:.3f}]")
        else:
            print()

        print(f"    xi_peak = {xi_peak[w]:.6e}")

        print()


def compute_muMax(s_array: npt.NDArray[np.float64],
                  mu_array: npt.NDArray[np.float64],
                  xi_array: npt.NDArray[np.float64],
                  mu_max: float,
                  base_path: str,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  delta_mu: float = 0.01
) -> npt.NDArray[np.float64]:

    s_unique = np.unique(s_array)
    xi_muMax = np.zeros(len(s_unique))

    for i, s in enumerate(s_unique):

        mask = (s_array == s) & (mu_array >= 0) & (mu_array <= mu_max)

        xi_vals = xi_array[mask]

        xi_muMax[i] = np.sum(xi_vals) * delta_mu / mu_max


    plt.figure(figsize=(8, 8), num=f"mu_max = {mu_max}")

    plt.plot(s_unique, (s_unique**2) * xi_muMax,
             label=fr"$\xi_{{[0,{mu_max}]}}$",
             linestyle='--', linewidth=0.6, marker='o', markersize=2)

    plt.title(fr"Scale dependance, $\mu_\mathrm{{max}} = {mu_max}$")

    plt.xlabel(r'$s \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(fr'$s^2 \xi_{{[0,{mu_max}]}} \,[h^{{-2}} \, \mathrm{{Mpc}}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # plt.savefig(f"{base_path}/muMax_{mu_max}.pdf", dpi=600)

    return xi_muMax


# ================================================================================================================================= #
#                                                                                                                                   #
#                                                        Projected function                                                         #
#                                                                                                                                   #
# ================================================================================================================================= #
def compute_projectedFunction(rp_array: npt.NDArray[np.float64],
                              xi_array: npt.NDArray[np.float64],
                              delta_pi: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Compute the projected funcition wₚ(rₚ) by integrating the correlation function ξ(rₚ, π)
    over uniformly spaced π-intervals.

    Parameters
    ----------
    rp_array : np.ndarray
        Array containing the perpendicular separation values rₚ for each data point.
    xi_array : np.ndarray
        Array of 2-point correlation function values ξ(rₚ, π) corresponding to each
        pair (rₚ, π).
    delta_pi : float, optional
        Width of each π-bin used in the discrete approximation of the integral.
        Default is `1.0`.

    Returns
    -------
    np.ndarray
        A 1D array of shape `(len(unique(rp_array))`.
    """
    
    rp_unique = np.unique(rp_array)
    wp_array = np.zeros(len(rp_unique))

    for i, rp in enumerate(rp_unique):

        mask = (rp_array == rp)
        xi_vals = xi_array[mask]

        wp_array[i] = 2.0 * np.sum(xi_vals) * delta_pi

    return wp_array


def plot_projectedFunction(rp_unique: npt.NDArray[np.float64],
                           wp: npt.NDArray[np.float64],
                           base_path: str,
                           err_wp: Optional[npt.NDArray[np.float64]] = None,
                           xlim: Optional[Tuple[float, float]] = None,
                           ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot the projected function as a function of the perpendicular scale rₚ and save
    the resulting figure to file.

    Parameters
    ----------
    rp_unique : np.ndarray
        Array of unique perpendicular separation values rₚ at which the
        projected function is evaluated.
    wp : np.ndarray
        Array of containing the projected function values.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_wp : np.ndarray, optional
        Error estimate associated with the projected function.
        If provided, error bars are included in the plot.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot named
        `"projFunc.pdf"` in the specified directory.
    """

    label=r"w_\mathrm{p}"

    plt.figure(figsize=(8, 8), num="Projected function")

    y = (rp_unique**2) * wp
    
    if err_wp is not None:
        err_y = (rp_unique**2) * err_wp
    else:
        err_y = None

    plt.errorbar(rp_unique, y, yerr=err_y, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(f"Projected function")
    
    plt.xlabel(r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$r_\mathrm{p}^2 w_\mathrm{p} \,[h^{-2} \, \mathrm{Mpc}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/projFunc.pdf", dpi=600)


def plot_projectedFunction_measVScorr(rp_unique: npt.NDArray[np.float64],
                                      wp_measured: npt.NDArray[np.float64],
                                      wp_correct: npt.NDArray[np.float64],
                                      base_path: str,
                                      err_wp_measured: Optional[npt.NDArray[np.float64]] = None,
                                      err_wp_correct: Optional[npt.NDArray[np.float64]] = None,
                                      xlim: Optional[Tuple[float, float]] = None,
                                      ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot the measured projected function together with the corresponding
    corrected projected function, and save the comparison figure to file.

    Parameters
    ----------
    rp_unique : np.ndarray
        Array of unique perpendicular separation values rₚ at which the
        projected function is evaluated.
    wp_measured : np.ndarray
        Measured projected function.
    wp_correct : np.ndarray
        Corrected projected function.
    base_path : str
        Directory in which the output figure will be saved.
    err_wp_measured : np.ndarray, optional
        Error estimates for the measured projected function. If provided, error bars
        are added to the measured curve.
    err_wp_correct : np.ndarray, optional
        Error estimates for the corrected projected function, used to add error bars.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    None
        The function generates and saves a plot comparing measured and
        corrected projected function as `"projFunc_measVScorr.pdf"` in
        the specified directory.
    """

    plt.figure(figsize=(8, 8), num="Projected function Measured VS Correct")
    
    
    label_measured = r"$w_\mathrm{p}^\mathrm{measured}$"
    label_correct  = r"$w_\mathrm{p}^\mathrm{correct}$"

    y_measured = (rp_unique**2) * wp_measured
    y_correct = (rp_unique**2) * wp_correct

    if err_wp_measured is not None:
        err_y_measured = (rp_unique**2) * err_wp_measured
    else:
        err_y_measured = None

    if err_wp_correct is not None:
        err_y_correct = (rp_unique**2) * err_wp_correct
    else:
        err_y_correct = None

    plt.errorbar(rp_unique, y_measured, yerr=err_y_measured, label=label_measured, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)
    plt.errorbar(rp_unique, y_correct, yerr=err_y_correct, label=label_correct, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)

    plt.title(f"Projected function Measured VS Correct")
    
    plt.xlabel(r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$r_\mathrm{p}^2 w_\mathrm{p} \,[h^{-2} \, \mathrm{Mpc}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{base_path}/projFunc_measVScorr.pdf", dpi=600)


def plot_projectedFunction_ratio(rp_unique: npt.NDArray[np.float64],
                                 wp_measured: npt.NDArray[np.float64],
                                 wp_correct: npt.NDArray[np.float64],
                                 base_path: str,
                                 err_wp_measured: Optional[npt.NDArray[np.float64]] = None,
                                 err_wp_correct: Optional[npt.NDArray[np.float64]] = None,
                                 xlim: Optional[Tuple[float, float]] = None,
                                 ylim: Optional[Tuple[float, float]] = None
) -> Tuple[
    npt.NDArray[np.float64],
    Optional[npt.NDArray[np.float64]]
]:
    """
    Plot the ratio between measured and correct projected function, compute
    the propagated uncertainties, and save the resulting figure to file.

    Parameters
    ----------
    rp_unique : np.ndarray
        Array of unique perpendicular separation values rₚ at which the
        projected function is evaluated.
    wp_measured : np.ndarray
        Measured projected function.
    wp_correct : np.ndarray
        Correct projected function.
    base_path : str
        Path to the directory in which the output figure will be saved.
    err_wp_measured : np.ndarray, optional
        Error estimates for the measured projected funtion. If provided together with
        `err_wp_correct`, error bars on the ratio are computed.
    err_wp_correct : np.ndarray, optional
        Error estimates for the correct projected function, used in the uncertainty
        propagation of the ratio.
    xlim : tuple of float, optional
        Limits for the x-axis of the plot.
    ylim : tuple of float, optional
        Limits for the y-axis of the plot.

    Returns
    -------
    tuple
        A pair `(wp_ratio, err_ratio)`, where:
        - `wp_ratio` is the element-wise ratio between measured and
          corrected projected function.
        - `err_ratio` contains the propagated uncertainties of the ratio, or
          `None` if the error inputs are not provided.
    """

    wp_ratio = np.divide(
        wp_measured,
        wp_correct,
        out=np.full_like(wp_measured, np.nan),
        where=(wp_correct != 0)
    )

    if err_wp_measured is not None and err_wp_correct is not None:
        
        term1 = np.divide(
            err_wp_measured, wp_correct,
            out=np.zeros_like(err_wp_measured),
            where=(wp_correct != 0)
        )

        term2 = np.divide(
            wp_measured * err_wp_correct, wp_correct**2,
            out=np.zeros_like(err_wp_measured),
            where=(wp_correct != 0)
        )

        err_ratio = np.sqrt(term1**2 + term2**2)

    else:
        err_ratio = None

    plt.figure(figsize=(8, 8), num="Ratio Measured VS Correct")

    
    label = r"$\frac{w_\mathrm{p}^\mathrm{measured}}{w_\mathrm{p}^\mathrm{correct}}$"

    y = wp_ratio
        
    if err_ratio is not None:
        err_y = err_ratio
    else:
        err_y = None

    plt.errorbar(rp_unique, y, yerr=err_y, label=label, linestyle='--', linewidth=0.6, marker='o', markersize=2, capsize=2)    

    plt.title(f"Projected function ratio")

    plt.xlabel(r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$\frac{w_\mathrm{p}^\mathrm{measured}}{w_\mathrm{p}^\mathrm{correct}}$')
    
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim) # this is necessary because when wedge_correct is near 0, the ratio explodes => so we limit the plot on y-axis

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{base_path}/projFunc_ratio.pdf", dpi=600)

    return wp_ratio, err_ratio


def compute_projectedFunction_BAOpeaks(rp_unique: npt.NDArray[np.float64],
                                wp: npt.NDArray[np.float64],
                                err_wp: Optional[npt.NDArray[np.float64]] = None,
                                rp_min: float = 50,
                                rp_max: float = 150
) -> Union[
    Tuple[float, float],
    Tuple[float, float, float, float]
]:
    """
    Identify the BAO peak position within a specified perpendicular separation 
    range for the projected correlation function wₚ(rₚ).

    Parameters
    ----------
    rp_unique : np.ndarray
        Array of unique perpendicular separation values rₚ at which the
        projected function is evaluated.
    wp : np.ndarray
        Array containing the projected function values.
    err_wp : np.ndarray, optional
        Array of error estimates associated with the projected function. If provided, 
        confidence intervals around the BAO peak position are computed.
    rp_min : float, optional
        Lower bound of the projected separation range in which the BAO peak is searched.
    rp_max : float, optional
        Upper bound of the projected separation range in which the BAO peak is searched.

    Returns
    -------
    tuple
        If no error estimates are provided:
            `(rp_peak, wp_peak)`
        where:
            - `rp_peak` contains the BAO peak position.
            - `wp_peak` contains the corresponding peak values.

        If errors are provided:
            `(rp_peak, wp_peak, rp_low, rp_high)`
        where:
            - `rp_low` and `rp_high` define the confidence interval bounds for
              each peak, obtained from the region where the projected function
              remains within 1σ of its maximum.
    """

    # restrict BAO window
    mask = (rp_unique >= rp_min) & (rp_unique <= rp_max)
    rp_BAO = rp_unique[mask]
    wp_BAO = wp[mask]

    # find the BAO peak
    peaks, _ = find_peaks(wp_BAO)

    if len(peaks) == 0:
        # fallback: global maximum inside the window
        idx = np.argmax(wp_BAO)
    else:
        # choose the highest local peak
        idx = peaks[np.argmax(wp_BAO[peaks])]

    rp_peak = rp_BAO[idx]
    w_peak = wp_BAO[idx]

    # return peak position if no errors are provided
    if err_wp is None:
        return rp_peak, w_peak

    # if errors are provided, compute confidence interval
    err_BAO = err_wp[mask]

    # confidence interval = all rp where wp(rp) stays within 1σ of the peak value
    threshold = w_peak - err_BAO[idx]

    mask_conf = wp_BAO >= threshold
    rp_conf = rp_BAO[mask_conf]

    if rp_conf.size == 0:
        rp_low = np.nan
        rp_high = np.nan
    else:
        rp_low = rp_conf.min()
        rp_high = rp_conf.max()

    return rp_peak, w_peak, rp_low, rp_high


def print_projectedFunction_BAOintervals(rp_peak: float,
                                  wp_peak: float,
                                  rp_low: Optional[float] = None,
                                  rp_high: Optional[float] = None
) -> None:
    """
    Print the BAO peak position for the projected function wₚ(rₚ),
    optionally including the corresponding confidence interval.

    Parameters
    ----------
    rp_peak : float
        BAO peak position.
    wp_peak : float
        Peak value of the projected function.
    rp_low : float, optional
        Lower bound of the confidence interval.
    rp_high : float, optional
        Upper bound of the confidence interval.

    Returns
    -------
    None
        The function prints the peak position and, if available, the
        confidence interval.
    """

    print(f"    rp_peak = {rp_peak:.3f}", end="")

    if rp_low is not None and rp_high is not None:
        print(f", rp confidence interval: [{rp_low:.3f}, {rp_high:.3f}]")
    else:
        print()

    print(f"    wp_peak = {wp_peak:.6e}")

    print()


def compute_piMax(rp_array: npt.NDArray[np.float64],
                  pi_array: npt.NDArray[np.float64],
                  xi_array: npt.NDArray[np.float64],
                  pi_max: float,
                  base_path: str,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  delta_pi: float = 1.0
) -> npt.NDArray[np.float64]:

    rp_unique = np.unique(rp_array)
    xi_piMax = np.zeros(len(rp_unique))

    for i, rp in enumerate(rp_unique):

        mask = (rp_array == rp) & (pi_array <= pi_max)

        xi_vals = xi_array[mask]

        xi_piMax[i] = np.sum(xi_vals) * delta_pi

    plt.figure(figsize=(8, 8), num=f"pi_max = {pi_max}")

    plt.plot(rp_unique, (rp_unique**2) * xi_piMax,
             label=fr"$\xi_{{[0,{pi_max}]}}$",
             linestyle='--', linewidth=0.6, marker='o', markersize=2)

    plt.title(fr"Scale dependance, $\pi_\mathrm{{max}} = {pi_max}$")

    plt.xlabel(r'$r_\mathrm{p} \,[h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(fr'$r_\mathrm{{p}}^2 \xi_{{[0,{pi_max}]}} \,[h^{{-2}} \, \mathrm{{Mpc}}^2]$')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # plt.savefig(f"{base_path}/piMax_{pi_max}.pdf", dpi=600)

    return xi_piMax