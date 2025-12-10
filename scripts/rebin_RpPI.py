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
rp_matrix_meas, pi_matrix_meas, dd_matrix_meas, dr_matrix_meas, rr_matrix_meas = read.readFITS_pairs_series_RpPI("data/measurements_pre_rec/m_z1_measured",
                                                                                                                 nElements, kind="measured")

print("Reading correct files...", end=" ")
rp_matrix_corr, pi_matrix_corr, dd_matrix_corr, dr_matrix_corr, rr_matrix_corr = read.readFITS_pairs_series_RpPI("data/measurements_pre_rec/m_z1_correct",
                                                                                                                 nElements, kind="correct")


# rebinning and compute 2PCF ------------------------------------------------------------------------------------------------------
# we want to reduce noise and we don't need to have the fine binning we have (200 bins in both rp and pi)
# so we rebin both rp and pi into 40 bins (5 Mpc/h per bin)
rpReb_matrix_meas, piReb_matrix_meas, ddReb_matrix_meas, drReb_matrix_meas, rrReb_matrix_meas = rebin.rebin_RpPI(rp_matrix= rp_matrix_meas,
                                                                                                                 pi_matrix= pi_matrix_meas,
                                                                                                                 dd_matrix= dd_matrix_meas,
                                                                                                                 dr_matrix= dr_matrix_meas,
                                                                                                                 rr_matrix= rr_matrix_meas,
                                                                                                                 delta_rp= 5, delta_pi= 5)

xiReb_matrix_meas = rebin.compute_xi(ddReb_matrix_meas, drReb_matrix_meas, rrReb_matrix_meas)


rpReb_matrix_corr, piReb_matrix_corr, ddReb_matrix_corr, drReb_matrix_corr, rrReb_matrix_corr = rebin.rebin_RpPI(rp_matrix= rp_matrix_corr,
                                                                                                                 pi_matrix= pi_matrix_corr,
                                                                                                                 dd_matrix= dd_matrix_corr,
                                                                                                                 dr_matrix= dr_matrix_corr,
                                                                                                                 rr_matrix= rr_matrix_corr,
                                                                                                                 delta_rp= 5, delta_pi= 5)

xiReb_matrix_corr = rebin.compute_xi(ddReb_matrix_corr, drReb_matrix_corr, rrReb_matrix_corr)


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
rpMean_array_meas = np.mean(rpReb_matrix_meas, axis=0) # this is probably useless
piMean_array_meas = np.mean(piReb_matrix_meas, axis=0) # just a check
xiMean_array_meas = np.mean(xiReb_matrix_meas, axis=0)
# xiStd_array_measured = np.std(xiReb_matrix_meas, axis=0)

rpMean_array_corr = np.mean(rpReb_matrix_corr, axis=0) # this is probably useless
piMean_array_corr = np.mean(piReb_matrix_corr, axis=0) # just a check
xiMean_array_corr = np.mean(xiReb_matrix_corr, axis=0)
# xiStd_array_correct = np.std(xiReb_matrix_corr, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(rpMean_array_meas, rpMean_array_corr, atol=1e-4) and \
    np.allclose(piMean_array_meas, piMean_array_corr, atol=1e-4): # this should always be true

    # plot measured
    RP, PI, _ = bm.plot_contourf(coords="RpPI",
                                 x_array=rpMean_array_meas,
                                 y_array=piMean_array_meas,
                                 z_array=xiMean_array_meas,
                                 base_path="graphs/measurements_pre_rec_rebin",
                                 kind="measured",
                                 xlim=(0,40),
                                 ylim=(0,40))
    
    # plot correct
    bm.plot_contourf(coords="RpPI",
                     x_array=rpMean_array_corr,
                     y_array=piMean_array_corr,
                     z_array=xiMean_array_corr,
                     base_path="graphs/measurements_pre_rec_rebin",
                     kind="correct",
                     xlim=(0,40),
                     ylim=(0,40))
    
else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")


# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
# plot in r_p and pi

xiMean_ratio = bm.plot_contourf_ratio(coords= "RpPI", 
                                      x_array= rpMean_array_meas,
                                      y_array= piMean_array_meas,
                                      z_array_measured= xiMean_array_meas,
                                      z_array_correct= xiMean_array_corr,
                                      v_min= -1.5,
                                      v_max= 1.5,
                                      base_path= "graphs/measurements_pre_rec_rebin",
                                      z_max= 2.1)

bm.plot_imshow_ratio(coords= "RpPI",
                     x_array= rpMean_array_meas,
                     y_array= piMean_array_meas,
                     z_array_measured= xiMean_array_meas,
                     z_array_correct= xiMean_array_corr,
                     xlim= (0,125),
                     ylim= (0,125),
                     v_min= -1.5,
                     v_max= 1.5,
                     interp="nearest",
                     base_path= "graphs/measurements_pre_rec_rebin",
                     z_max= 2.1)


# projected function --------------------------------------------------------------------------------------------------------------
print("\n===== Calculating projected function =====")

rp_unique = np.unique(rpMean_array_corr)

# we need to compute projected function but we dont do projFunc(mean) we do mean(projFunc) to have an accurate estimate of the projFunc
wp_measured_all = []
for i in range(nFiles):
    wp_i = prj.compute_projectedFunction(
        rpReb_matrix_meas[i],
        xiReb_matrix_meas[i],
        delta_pi=5.0
    )
    wp_measured_all.append(wp_i)
wp_measured_all = np.array(wp_measured_all) # shape = (1000, n_rp_unique)

wpMean_measured = wp_measured_all.mean(axis=0)
wpStd_measured = wp_measured_all.std(axis=0)

wp_correct_all = []
for i in range(nFiles):
    wp_i = prj.compute_projectedFunction(
        rpReb_matrix_corr[i],
        xiReb_matrix_corr[i],
        delta_pi=5.0
    )
    wp_correct_all.append(wp_i)
wp_correct_all = np.array(wp_correct_all) # shape = (1000, n_rp_unique)

wpMean_correct = wp_correct_all.mean(axis=0)
wpStd_correct = wp_correct_all.std(axis=0)

prj.plot_projectedFunction_measVScorr(rp_unique= rp_unique,
                                      wp_measured= wpMean_measured,
                                      wp_correct= wpMean_correct,
                                      err_wp_measured= wpStd_measured,
                                      err_wp_correct= wpStd_correct,
                                      base_path= "graphs/measurements_pre_rec_rebin")

ratio, errRatio = prj.plot_projectedFunction_ratio(rp_unique= rp_unique,
                                                   wp_measured= wpMean_measured,
                                                   wp_correct= wpMean_correct,
                                                   err_wp_measured= wpStd_measured,
                                                   err_wp_correct= wpStd_correct,
                                                   base_path= "graphs/measurements_pre_rec_rebin",
                                                   ylim=(0,2))

rp_peak_measured, wp_peak_measured, rp_low_measured, rp_high_measured = prj.compute_projectedFunction_BAOpeaks(rp_unique= rp_unique, # type: ignore
                                                                                                               wp= wpMean_measured,
                                                                                                               err_wp= wpStd_measured,
                                                                                                               rp_min= 80,
                                                                                                               rp_max= 120)

print("Printing measured BAO peak...")
prj.print_projectedFunction_BAOintervals(rp_peak= rp_peak_measured,
                                         wp_peak= wp_peak_measured,
                                         rp_low= rp_low_measured,
                                         rp_high= rp_high_measured)

rp_peak_correct, wp_peak_correct, rp_low_correct, rp_high_correct = prj.compute_projectedFunction_BAOpeaks(rp_unique= rp_unique, # type: ignore
                                                                                                           wp= wpMean_correct,
                                                                                                           err_wp= wpStd_correct,
                                                                                                           rp_min= 80,
                                                                                                           rp_max= 120)

print("Printing correct BAO peak...")
prj.print_projectedFunction_BAOintervals(rp_peak= rp_peak_correct,
                                         wp_peak= wp_peak_correct,
                                         rp_low= rp_low_correct,
                                         rp_high= rp_high_correct)

# we integrate xi(r_p, pi) from pi=0 to pi=pi_max
# in order to find the pi_max corresponding to the beginning of the scale dependance in r_p
prj.compute_piMax(rpMean_array_meas, piMean_array_meas, xiMean_array_meas, 60, "graphs/measurements_pre_rec_rebin")


plt.show()