import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
from libs import BiMaps as bm
from libs import Projections as prj
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
rp_matrix_measured, pi_matrix_measured, xi_matrix_measured = read.readFITS_auto_series_RpPI(
    "data/measurements_pre_rec/m_z1_measured",
    nElements,
    kind="measured"
)

print("Reading correct files...", end=" ")
rp_matrix_correct, pi_matrix_correct, xi_matrix_correct = read.readFITS_auto_series_RpPI(
    "data/measurements_pre_rec/m_z1_correct",
    nElements,
    kind="correct"
)

xi_matrix_ratio = xi_matrix_measured / xi_matrix_correct

# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
rpMean_array_measured = np.mean(rp_matrix_measured, axis=0) # this is probably useless
piMean_array_measured = np.mean(pi_matrix_measured, axis=0) # just a check
xiMean_array_measured = np.mean(xi_matrix_measured, axis=0)

rpMean_array_correct = np.mean(rp_matrix_correct, axis=0) # this is probably useless
piMean_array_correct = np.mean(pi_matrix_correct, axis=0) # just a check
xiMean_array_correct = np.mean(xi_matrix_correct, axis=0)

xiMean_array_ratio = np.mean(xi_matrix_ratio, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(rpMean_array_measured, rpMean_array_correct, atol=1e-4) and \
    np.allclose(piMean_array_measured, piMean_array_correct, atol=1e-4): # this should always be true

    vmin = min(xiMean_array_measured.min(), xiMean_array_correct.min())
    vmax = max(xiMean_array_measured.max(), xiMean_array_correct.max())

    # plot measured
    RP, PI, _ = bm.plot_contourf(
        coords="RpPI",
        x_array=rpMean_array_measured,
        y_array=piMean_array_measured,
        z_array=xiMean_array_measured,
        base_path="graphs/measurements_pre_rec",
        kind="measured",
        v_min= vmin,
        v_max= vmax
    )
    
    bm.plot_imshow(
        coords="RpPI",
        x_array=rpMean_array_measured,
        y_array=piMean_array_measured,
        z_array=xiMean_array_measured,
        base_path="graphs/measurements_pre_rec",
        kind="measured",
        v_min= vmin,
        v_max= vmax
    )
    
    # plot correct
    bm.plot_contourf(
        coords="RpPI",
        x_array=rpMean_array_correct,
        y_array=piMean_array_correct,
        z_array=xiMean_array_correct,
        base_path="graphs/measurements_pre_rec",
        kind="correct",
        v_min= vmin,
        v_max= vmax
    )
    
    bm.plot_imshow(
        coords="RpPI",
        x_array=rpMean_array_correct,
        y_array=piMean_array_correct,
        z_array=xiMean_array_correct,
        base_path="graphs/measurements_pre_rec",
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
    x_array= rpMean_array_measured,
    y_array= piMean_array_measured,
    z_array_ratio= xiMean_array_ratio,
    v_min= -1.5,
    v_max= 1.5,
    base_path= "graphs/measurements_pre_rec",
    z_max= 2.1
)

bm.plot_imshow_ratio(
    coords= "RpPI",
    x_array= rpMean_array_measured,
    y_array= piMean_array_measured,
    z_array_ratio= xiMean_array_ratio,
    v_min= -1.5,
    v_max= 1.5,
    base_path= "graphs/measurements_pre_rec",
    z_max= 2.1
)


# projected function --------------------------------------------------------------------------------------------------------------
print("\n===== Calculating projected function =====")

rp_unique = np.unique(rpMean_array_correct)

# we need to compute projected function but we dont do projFunc(mean)
# we do mean(projFunc) to have an accurate estimate of the projFunc
wp_measured_all = []
wp_correct_all = []
wp_ratio_all = []
for i in range(nFiles):
    wp_i_measured = prj.compute_projectedFunction(
        rp_matrix_measured[i],
        xi_matrix_measured[i]
    )
    wp_measured_all.append(wp_i_measured)

    wp_i_correct = prj.compute_projectedFunction(
        rp_matrix_correct[i],
        xi_matrix_correct[i]
    )
    wp_correct_all.append(wp_i_correct)

    wp_i_ratio = wp_i_measured / wp_i_correct
    wp_ratio_all.append(wp_i_ratio)

wp_measured_all = np.array(wp_measured_all) # shape = (1000, n_unique)
wpMean_measured = wp_measured_all.mean(axis=0)
wpStd_measured = wp_measured_all.std(axis=0, ddof=1)

wp_correct_all = np.array(wp_correct_all) # shape = (1000, n_unique)
wpMean_correct = wp_correct_all.mean(axis=0)
wpStd_correct = wp_correct_all.std(axis=0, ddof=1)

wp_ratio_all = np.array(wp_ratio_all) # shape = (1000, n_unique)
wpMean_ratio = wp_ratio_all.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
wpStd_ratio = wp_ratio_all.std(axis=0, ddof=1) # and we want to be the most accurate possible => we need mean(ratio) [same for std]

prj.plot_projectedFunction_measVScorr(
    rp_unique= rp_unique,
    wp_measured= wpMean_measured,
    wp_correct= wpMean_correct,
    err_wp_measured= wpStd_measured,
    err_wp_correct= wpStd_correct,
    base_path= "graphs/measurements_pre_rec"
)

prj.plot_projectedFunction_ratio(
    rp_unique= rp_unique,
    wp_ratio= wpMean_ratio,
    err_wp_ratio= wpStd_ratio,
    base_path= "graphs/measurements_pre_rec",
    ylim=(-2,2)
)

rp_peak_measured, wp_peak_measured, rp_low_measured, rp_high_measured = prj.compute_projectedFunction_BAOpeaks( # type: ignore
    rp_unique= rp_unique,
    wp= wpMean_measured,
    err_wp= wpStd_measured,
    rp_min= 80,
    rp_max= 120
)

print("Printing measured BAO peak...")
prj.print_projectedFunction_BAOintervals(
    rp_peak= rp_peak_measured,
    wp_peak= wp_peak_measured,
    rp_low= rp_low_measured,
    rp_high= rp_high_measured
)

rp_peak_correct, wp_peak_correct, rp_low_correct, rp_high_correct = prj.compute_projectedFunction_BAOpeaks( # type: ignore
    rp_unique= rp_unique,
    wp= wpMean_correct,
    err_wp= wpStd_correct,
    rp_min= 80,
    rp_max= 120
)

print("Printing correct BAO peak...")
prj.print_projectedFunction_BAOintervals(
    rp_peak= rp_peak_correct,
    wp_peak= wp_peak_correct,
    rp_low= rp_low_correct,
    rp_high= rp_high_correct
)

# we integrate xi(r_p, pi) from pi=0 to pi=pi_max
# in order to find the pi_max corresponding to the beginning of the scale dependance in r_p
prj.compute_piMax(
    rp_matrix= rp_matrix_measured,
    pi_matrix= pi_matrix_measured,
    xi_matrix= xi_matrix_measured,
    pi_max_values= np.array([100, 110, 120, 130, 140]),
    base_path= "graphs/measurements_pre_rec"
)

prj.compute_piMax_ratio(
    rp_matrix= rp_matrix_measured,
    pi_matrix= pi_matrix_measured,
    xi_matrix_measured= xi_matrix_measured,
    xi_matrix_correct= xi_matrix_correct,
    pi_max_values= np.array([100, 110, 120, 130, 140]),
    base_path= "graphs/measurements_pre_rec",
    ylim= (0,2)
)


plt.show()
plt.close('all')