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
                       xi_array: npt.NDArray[np.float64]
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
    delta_mu = np.float64(0.01)
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