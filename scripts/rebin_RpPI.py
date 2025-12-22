import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
from libs import BiMaps as bm
from libs import Projections as prj
from libs import Rebinning as rebin
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = True  # use real LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

try:
    matplotlib.use("QtAgg")
except ImportError:
    print("QtAgg backend not available, using default backend.")
matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


# read FITS files -----------------------------------------------------------------------------------------------------------------
nElements = 40000
nFiles = 200

print("\nReading measured files...", end=" ")
rp_matrix_meas, pi_matrix_meas, dd_matrix_meas, dr_matrix_meas, rr_matrix_meas, Ndd_matrix_meas, Ndr_matrix_meas, Nrr_matrix_meas = \
    read.readFITS_pairs_series_RpPI(
        "data/measurements_pre_rec/m_z1_measured",
        nElements,
        kind="measured"
    )

print("Reading correct files...", end=" ")
rp_matrix_corr, pi_matrix_corr, dd_matrix_corr, dr_matrix_corr, rr_matrix_corr, Ndd_matrix_corr, Ndr_matrix_corr, Nrr_matrix_corr = \
    read.readFITS_pairs_series_RpPI(
        "data/measurements_pre_rec/m_z1_correct",
        nElements,
        kind="correct"
    )

# this is for the errors: std computes the error of the single measure,
# but, we when we do more realisations, we actually need to compute the error of the mean = error / sqrt(nFiles)
sigmaMean = True
if sigmaMean:
    normaliz = 1.0 / np.sqrt(nFiles)
else:
    normaliz = 1.0


# rebinning and compute 2PCF ------------------------------------------------------------------------------------------------------
# we want to reduce noise and we don't need to have the fine binning we have (200 bins in both rp and pi)
# so we rebin both rp and pi into 40 bins (5 Mpc/h per bin)
xi_matrix_meas = rebin.compute_xi(
    dd= dd_matrix_meas,
    dr= dr_matrix_meas,
    rr= rr_matrix_meas,
    Ndd= Ndd_matrix_meas,
    Ndr= Ndr_matrix_meas,
    Nrr= Nrr_matrix_meas
)

xi_matrix_corr = rebin.compute_xi(
    dd= dd_matrix_corr,
    dr= dr_matrix_corr,
    rr= rr_matrix_corr,
    Ndd= Ndd_matrix_corr,
    Ndr= Ndr_matrix_corr,
    Nrr= Nrr_matrix_corr
)

xi_matrix_ratio = xi_matrix_meas / xi_matrix_corr

rp_matrix_meas_reb, pi_matrix_meas_reb, dd_matrix_meas_reb, dr_matrix_meas_reb, rr_matrix_meas_reb = \
    rebin.rebin_RpPI(
        rp_matrix= rp_matrix_meas,
        pi_matrix= pi_matrix_meas,
        dd_matrix= dd_matrix_meas,
        dr_matrix= dr_matrix_meas,
        rr_matrix= rr_matrix_meas,
        delta_rp= 5, delta_pi= 5
    )

xi_matrix_meas_reb = rebin.compute_xi(
    dd= dd_matrix_meas_reb,
    dr= dr_matrix_meas_reb,
    rr= rr_matrix_meas_reb,
    Ndd= Ndd_matrix_meas,
    Ndr= Ndr_matrix_meas,
    Nrr= Nrr_matrix_meas
)

rp_matrix_corr_reb, pi_matrix_corr_reb, dd_matrix_corr_reb, dr_matrix_corr_reb, rr_matrix_corr_reb = \
    rebin.rebin_RpPI(
        rp_matrix= rp_matrix_corr,
        pi_matrix= pi_matrix_corr,
        dd_matrix= dd_matrix_corr,
        dr_matrix= dr_matrix_corr,
        rr_matrix= rr_matrix_corr,
        delta_rp= 5, delta_pi= 5
    )

xi_matrix_corr_reb = rebin.compute_xi(
    dd= dd_matrix_corr_reb,
    dr= dr_matrix_corr_reb,
    rr= rr_matrix_corr_reb,
    Ndd= Ndd_matrix_corr,
    Ndr= Ndr_matrix_corr,
    Nrr= Nrr_matrix_corr
)

xi_matrix_ratio_reb = xi_matrix_meas_reb / xi_matrix_corr_reb


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
# not rebinned ones
rpMean_array_meas = np.mean(rp_matrix_meas, axis=0) # this is probably useless
piMean_array_meas = np.mean(pi_matrix_meas, axis=0) # just a check
xiMean_array_meas = np.mean(xi_matrix_meas, axis=0)

rpMean_array_corr = np.mean(rp_matrix_corr, axis=0) # this is probably useless
piMean_array_corr = np.mean(pi_matrix_corr, axis=0) # just a check
xiMean_array_corr = np.mean(xi_matrix_corr, axis=0)

xiMean_array_ratio = np.mean(xi_matrix_ratio, axis=0)

# rebinned ones
rpMean_array_meas_reb = np.mean(rp_matrix_meas_reb, axis=0) # this is probably useless
piMean_array_meas_reb = np.mean(pi_matrix_meas_reb, axis=0) # just a check
xiMean_array_meas_reb = np.mean(xi_matrix_meas_reb, axis=0)

rpMean_array_corr_reb = np.mean(rp_matrix_corr_reb, axis=0) # this is probably useless
piMean_array_corr_reb = np.mean(pi_matrix_corr_reb, axis=0) # just a check
xiMean_array_corr_reb = np.mean(xi_matrix_corr_reb, axis=0)

xiMean_array_ratio_reb = np.mean(xi_matrix_ratio_reb, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(rpMean_array_meas_reb, rpMean_array_corr_reb, atol=1e-4) and \
    np.allclose(piMean_array_meas_reb, piMean_array_corr_reb, atol=1e-4): # this should always be true

    vmin = min(xiMean_array_meas_reb.min(), xiMean_array_corr_reb.min())
    vmax = max(xiMean_array_meas_reb.max(), xiMean_array_corr_reb.max())

    # plot measured
    RP, PI, _ = bm.plot_contourf(
        coords="RpPI",
        x_array=rpMean_array_meas_reb,
        y_array=piMean_array_meas_reb,
        z_array=xiMean_array_meas_reb,
        base_path="graphs/measurements_pre_rec_rebin",
        kind="measured",
        v_min= vmin,
        v_max= vmax
    )
    
    # plot correct
    bm.plot_contourf(
        coords="RpPI",
        x_array=rpMean_array_corr_reb,
        y_array=piMean_array_corr_reb,
        z_array=xiMean_array_corr_reb,
        base_path="graphs/measurements_pre_rec_rebin",
        kind="correct",
        v_min= vmin,
        v_max= vmax
    )
    
else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")


# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
bm.plot_contourf_ratio(
    coords= "RpPI",
    x_array= rpMean_array_meas_reb,
    y_array= piMean_array_meas_reb,
    z_array_ratio= xiMean_array_ratio_reb,
    v_min= -1.5,
    v_max= 1.5,
    base_path= "graphs/measurements_pre_rec_rebin",
    z_max= 2.1
)

bm.plot_imshow_ratio(
    coords= "RpPI",
    x_array= rpMean_array_meas_reb,
    y_array= piMean_array_meas_reb,
    z_array_ratio= xiMean_array_ratio_reb,
    v_min= -1.5,
    v_max= 1.5,
    interp="nearest",
    base_path= "graphs/measurements_pre_rec_rebin",
    z_max= 2.1
)

plt.show()
plt.close('all')


# projected function --------------------------------------------------------------------------------------------------------------
print("\n===== Calculating projected function =====")

rp_unique_reb = np.unique(rpMean_array_meas_reb)

# we need to compute projected function, but we dont do projFunc(mean)
# we do mean(projFunc) to have an accurate estimate of the projFunc
wp_measured_all_reb = []
wp_correct_all_reb = []
wp_ratio_all_reb = []
for i in range(nFiles):
    wp_i_measured_reb = prj.compute_projectedFunction(
        rp_matrix_meas_reb[i],
        xi_matrix_meas_reb[i]
    )
    wp_measured_all_reb.append(wp_i_measured_reb)

    wp_i_correct_reb = prj.compute_projectedFunction(
        rp_matrix_corr_reb[i],
        xi_matrix_corr_reb[i]
    )
    wp_correct_all_reb.append(wp_i_correct_reb)

    wp_i_ratio_reb = wp_i_measured_reb / wp_i_correct_reb
    wp_ratio_all_reb.append(wp_i_ratio_reb)

wp_measured_all_reb = np.array(wp_measured_all_reb) # shape = (1000, n_rp_unique)
wpMean_measured_reb = wp_measured_all_reb.mean(axis=0)
wpStd_measured_reb = wp_measured_all_reb.std(axis=0, ddof=1) * normaliz

wp_correct_all_reb = np.array(wp_correct_all_reb) # shape = (1000, n_rp_unique)
wpMean_correct_reb = wp_correct_all_reb.mean(axis=0)
wpStd_correct_reb = wp_correct_all_reb.std(axis=0, ddof=1) * normaliz

wp_ratio_all_reb = np.array(wp_ratio_all_reb) # shape = (1000, n_rp_unique)
wpMean_ratio_reb = wp_ratio_all_reb.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
wpStd_ratio_reb = wp_ratio_all_reb.std(axis=0, ddof=1) * normaliz # and we want to be the most accurate possible => we need mean(ratio)

# covariance matrices
wpCov_measured_reb = np.cov(wp_measured_all_reb, rowvar=False) # shape = (n_rp_unique, n_rp_unique)
wpCov_correct_reb = np.cov(wp_correct_all_reb, rowvar=False) # shape = (n_rp_unique, n_rp_unique)

label = r"Correlation Matrix for $w_\mathrm{p}$"

bm.plot_correlationMatrix(
    coords= "RpPI",
    scale= rp_unique_reb,
    cov= wpCov_measured_reb,
    base_path= "graphs/measurements_pre_rec_rebin",
    kind= "measured",
    fig= label,
    filename= "wpCov_meas"
)

bm.plot_correlationMatrix(
    coords= "RpPI",
    scale= rp_unique_reb,
    cov= wpCov_correct_reb,
    base_path= "graphs/measurements_pre_rec_rebin",
    kind= "correct",
    fig= label,
    filename= "wpCov_corr"
)

# plotting
prj.plot_projectedFunction_measVScorr(
    rp_unique= rp_unique_reb,
    wp_measured= wpMean_measured_reb,
    wp_correct= wpMean_correct_reb,
    err_wp_measured= wpStd_measured_reb,
    err_wp_correct= wpStd_correct_reb,
    base_path= "graphs/measurements_pre_rec_rebin"
)

prj.plot_projectedFunction_ratio(
    rp_unique= rp_unique_reb,
    wp_ratio= wpMean_ratio_reb,
    err_wp_ratio= wpStd_ratio_reb,
    base_path= "graphs/measurements_pre_rec_rebin",
    ylim=(-2,2)
)

# BAO peak finding
rp_peak_measured_reb, wp_peak_measured_reb, rp_low_measured_reb, rp_high_measured_reb = prj.compute_projectedFunction_BAOpeaks( # type: ignore
    rp_unique= rp_unique_reb,
    wp= wpMean_measured_reb,
    err_wp= wpStd_measured_reb,
    rp_min= 80,
    rp_max= 120
)

print("Printing measured BAO peak...")
prj.print_projectedFunction_BAOintervals(
    rp_peak= rp_peak_measured_reb,
    wp_peak= wp_peak_measured_reb,
    rp_low= rp_low_measured_reb,
    rp_high= rp_high_measured_reb
)

rp_peak_correct_reb, wp_peak_correct_reb, rp_low_correct_reb, rp_high_correct_reb = prj.compute_projectedFunction_BAOpeaks( # type: ignore
    rp_unique= rp_unique_reb,
    wp= wpMean_correct_reb,
    err_wp= wpStd_correct_reb,
    rp_min= 80,
    rp_max= 120
)

print("Printing correct BAO peak...")
prj.print_projectedFunction_BAOintervals(
    rp_peak= rp_peak_correct_reb,
    wp_peak= wp_peak_correct_reb,
    rp_low= rp_low_correct_reb,
    rp_high= rp_high_correct_reb
)

# we integrate xi(r_p, pi) from pi=0 to pi=pi_max
# in order to find the pi_max corresponding to the beginning of the scale dependance in r_p
prj.compute_piMax(
    rp_matrix= rp_matrix_meas_reb,
    pi_matrix= pi_matrix_meas_reb,
    xi_matrix= xi_matrix_meas_reb,
    pi_max_values= np.array([100, 110, 120, 130, 140]),
    base_path= "graphs/measurements_pre_rec_rebin",
    delta_pi= 5.0
)

prj.compute_piMax_ratio(
    rp_matrix= rp_matrix_meas_reb,
    pi_matrix= pi_matrix_meas_reb,
    xi_matrix_measured= xi_matrix_meas_reb,
    xi_matrix_correct= xi_matrix_corr_reb,
    pi_max_values= np.array([100, 110, 120, 130, 140]),
    base_path= "graphs/measurements_pre_rec_rebin",
    delta_pi= 5.0,
    ylim=(0,2)
)


plt.show()
plt.close('all')