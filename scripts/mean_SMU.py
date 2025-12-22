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
nFiles = 1000
nElements = 40000

print(f"\nReading {nFiles} measured files...")
s_matrix_measured, mu_matrix_measured, xi_matrix_measured = read.readFITS_auto_series_SMU(
    "data/z1_data/z1_measured",
    nFiles,
    nElements
)

print(f"Reading {nFiles} correct files...")
s_matrix_correct, mu_matrix_correct, xi_matrix_correct = read.readFITS_auto_series_SMU(
    "data/z1_data/z1_correct",
    nFiles,
    nElements
)

xi_matrix_ratio = xi_matrix_measured / xi_matrix_correct

print()

# this is for the errors: std computes the error of the single measure,
# but, we when we do more realisations, we actually need to compute the error of the mean = error / sqrt(nFiles)
sigmaMean = True
if sigmaMean:
    normaliz = 1.0 / np.sqrt(nFiles)
else:
    normaliz = 1.0


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
sMean_array_measured = np.mean(s_matrix_measured, axis=0) # this is probably useless
muMean_array_measured = np.mean(mu_matrix_measured, axis=0) # just a check
xiMean_array_measured = np.mean(xi_matrix_measured, axis=0)

sMean_array_correct = np.mean(s_matrix_correct, axis=0) # this is probably useless
muMean_array_correct = np.mean(mu_matrix_correct, axis=0) # just a check
xiMean_array_correct = np.mean(xi_matrix_correct, axis=0)

xiMean_array_ratio = np.mean(xi_matrix_ratio, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(muMean_array_measured, muMean_array_correct, atol=1e-4) and \
    np.allclose(sMean_array_measured, sMean_array_correct, atol=1e-4): # this should always be true

    vmin = min(xiMean_array_measured.min(), xiMean_array_correct.min())
    vmax = max(xiMean_array_measured.max(), xiMean_array_correct.max())

    # plot measured
    MU, S, _ = bm.plot_contourf(
        coords= "SMU",
        x_array= muMean_array_measured,
        y_array= sMean_array_measured,
        z_array= xiMean_array_measured,
        base_path= "graphs/z1",
        kind= "measured",
        draw_lines=False,
        v_min= vmin,
        v_max= vmax
    )

    # plot correct
    bm.plot_contourf(
        coords= "SMU",
        x_array= muMean_array_correct,
        y_array= sMean_array_correct,
        z_array= xiMean_array_correct,
        base_path= "graphs/z1",
        kind="correct",
        draw_lines=False,
        v_min= vmin,
        v_max= vmax
    )

else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")


# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
# plot in s and mu
bm.plot_contourf_ratio(
    coords= "SMU",
    x_array= muMean_array_measured,
    y_array= sMean_array_measured,
    z_array_ratio= xiMean_array_ratio,
    base_path= "graphs/z1",
    v_min= -1.5,
    v_max= 1.5,
    z_max= 2
)

bm.plot_imshow_ratio(
    coords= "SMU",
    x_array= muMean_array_measured,
    y_array= sMean_array_measured,
    z_array_ratio= xiMean_array_ratio,
    base_path= "graphs/z1",
    v_min= -1.5,
    v_max= 1.5,
    z_max= 2
)

# plot in r_p and pi
# RP = S * np.sqrt(1 - MU**2)
# PI = S * MU

# plt.figure(figsize=(9,8), num="Ratio (r_p, pi)")
# contourRpPI = plt.contourf(RP, PI, np.log(np.abs(XI_RATIO)), levels=20, cmap='turbo') # we use log to better observe differences in levels
#                                                                                       # abs is for avoiding log of negative numbers

# cbarRpPI = plt.colorbar(contourRpPI, label=r'$\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}$')
# xi_interpol_ticks = np.linspace(np.min(np.log(np.abs(xi_ratio))), np.max(np.log(np.abs(xi_ratio))), 9)
# cbarRpPI.set_ticks(xi_interpol_ticks)
# cbarRpPI.set_ticklabels([f"{tick:.3f}" for tick in xi_interpol_ticks])

# rp_ticks = np.linspace(np.min(RP), np.max(RP), 10)
# pi_ticks = np.linspace(np.min(PI), np.max(PI), 10)
# plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
# plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

# # plt.xlim(0,30)
# # plt.ylim(-15,15)

# plt.xlabel(r'$r_p \,(h^{-1} \, \mathrm{Mpc})$')
# plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
# plt.title(r'2D map of $\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}(r_p, \pi)$')
# plt.tight_layout()
# plt.savefig("graphs/z1/2DmapRpPI_ratio.pdf", dpi=600)

plt.show()
plt.close('all')


# custering wedges ----------------------------------------------------------------------------------------------------------------
# clustering wedge (s) = (int_(mu_min)^(mu_max) xi(s,mu) dmu) / (mu_max - mu_min)
# for custering wedges we often use 0 < mu < 1
# however we have -1 < mu < 1
# but for galaxies pairs the correlation function should be symmetrical xi(s,mu) = xi(s,-mu) --> I verified it's true

nWedges = 2
print(f"\n===== Calculating {nWedges} clustering wedges =====")

s_unique = np.unique(sMean_array_correct)

# we need to compute clustering wedges but we dont do wedge(mean) we do mean(wedges) to have an accurate estimate of the wedges
wedges_measured_all = []
wedges_correct_all = []
wedges_ratio_all = []
for i in range(nFiles):
    wedges_i_measured = prj.compute_clusteringWedges(
        nWedges,
        s_matrix_measured[i],
        mu_matrix_measured[i],
        xi_matrix_measured[i]
    )
    wedges_measured_all.append(wedges_i_measured)

    wedges_i_correct = prj.compute_clusteringWedges(
        nWedges,
        s_matrix_correct[i],
        mu_matrix_correct[i],
        xi_matrix_correct[i]
    )
    wedges_correct_all.append(wedges_i_correct)

    wedges_i_ratio = wedges_i_measured / wedges_i_correct
    wedges_ratio_all.append(wedges_i_ratio)

wedges_measured_all = np.array(wedges_measured_all) # shape = (1000, nWedges, n_s_unique)
wedgesMean_measured = wedges_measured_all.mean(axis=0)
wedgesStd_measured = wedges_measured_all.std(axis=0, ddof=1) * normaliz

wedges_correct_all = np.array(wedges_correct_all) # shape = (1000, nWedges, n_s_unique)
wedgesMean_correct = wedges_correct_all.mean(axis=0)
wedgesStd_correct = wedges_correct_all.std(axis=0, ddof=1) * normaliz

wedges_ratio_all = np.array(wedges_ratio_all) # shape = (1000, nWedges, n_s_unique)
wedgesMean_ratio = wedges_ratio_all.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
wedgesStd_ratio = wedges_ratio_all.std(axis=0, ddof=1) * normaliz # and we want to be the most accurate possible => we need mean(ratio)

prj.plot_clusteringWedges_measVScorr(
    n_wedges= nWedges,
    s_unique= s_unique,
    wedges_measured= wedgesMean_measured,
    wedges_correct= wedgesMean_correct,
    err_wedges_measured= wedgesStd_measured,
    err_wedges_correct= wedgesStd_correct,
    base_path= "graphs/z1"
)

prj.plot_clusteringWedges_ratio(
    n_wedges= nWedges,
    s_unique= s_unique,
    wedges_ratio= wedgesMean_ratio,
    err_wedges_ratio= wedgesStd_ratio,
    base_path= "graphs/z1",
    ylim=(0,2)
)

s_peak_array_measured, xi_peak_array_measured, s_low_measured, s_high_measured = prj.compute_clusteringWedges_BAOpeaks( # type: ignore
    n_wedges= nWedges,
    s_unique= s_unique,
    wedges= wedgesMean_measured,
    err_wedges= wedgesStd_measured,
    s_min= 85,
    s_max= 115
)

print("Printing measured BAO peak...")
prj.print_clusteringWedges_BAOintervals(
    n_wedges= nWedges,
    s_peak= s_peak_array_measured,
    xi_peak= xi_peak_array_measured,
    s_low= s_low_measured,
    s_high= s_high_measured
)

s_peak_array_correct, xi_peak_array_correct, s_low_correct, s_high_correct = prj.compute_clusteringWedges_BAOpeaks( # type: ignore
    n_wedges= nWedges,
    s_unique= s_unique,
    wedges= wedgesMean_correct,
    err_wedges= wedgesStd_correct,
    s_min= 85,
    s_max= 115
)

print("Printing correct BAO peak...")
prj.print_clusteringWedges_BAOintervals(
    n_wedges= nWedges,
    s_peak= s_peak_array_correct,
    xi_peak= xi_peak_array_correct,
    s_low= s_low_correct,
    s_high= s_high_correct
)

# we integrate xi(mu, s) from mu=0 to mu=mu_max
# in order to find the mu_max corresponding to the beginning of the scale dependance in s
prj.compute_muMax(
    s_matrix= s_matrix_measured,
    mu_matrix= mu_matrix_measured,
    xi_matrix= xi_matrix_measured,
    mu_max_values= np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    base_path= "graphs/z1"
)

prj.compute_muMax_ratio(
    s_matrix= s_matrix_measured,
    mu_matrix= mu_matrix_measured,
    xi_matrix_measured= xi_matrix_measured,
    xi_matrix_correct= xi_matrix_correct,
    mu_max_values= np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    base_path= "graphs/z1",
    ylim=(0.6,0.8)
)

plt.show()
plt.close('all')


# multipoles projection -----------------------------------------------------------------------------------------------------------
# same thing for the multipoles
lValues = np.array([0])
print("\n===== Calculating monopoles =====")

multipoles_measured_all = []
multipoles_correct_all = []
multipoles_ratio_all = []
for i in range(nFiles):
    multipoles_i_measured = prj.compute_multipoles(
        lValues,
        s_matrix_measured[i],
        mu_matrix_measured[i],
        xi_matrix_measured[i]
    )
    multipoles_measured_all.append(multipoles_i_measured)

    multipoles_i_correct = prj.compute_multipoles(
        lValues,
        s_matrix_correct[i],
        mu_matrix_correct[i],
        xi_matrix_correct[i]
    )
    multipoles_correct_all.append(multipoles_i_correct)

    multipoles_i_ratio = multipoles_i_measured / multipoles_i_correct
    multipoles_ratio_all.append(multipoles_i_ratio)

multipoles_measured_all = np.array(multipoles_measured_all) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_measured = multipoles_measured_all.mean(axis=0)
multipolesStd_measured = multipoles_measured_all.std(axis=0, ddof=1) * normaliz

multipoles_correct_all = np.array(multipoles_correct_all) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_correct = multipoles_correct_all.mean(axis=0)
multipolesStd_correct = multipoles_correct_all.std(axis=0, ddof=1) * normaliz

multipoles_ratio_all = np.array(multipoles_ratio_all) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_ratio = multipoles_ratio_all.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
multipolesStd_ratio = multipoles_ratio_all.std(axis=0, ddof=1) * normaliz # and we want to be the most accurate possible => we need mean(ratio)

nPoints = 5
print(f"First {nPoints} measured points:")
prj.print_multipoles(
    l_values= lValues,
    s_array= sMean_array_measured,
    xi_multipoles= multipolesMean_measured,
    n_values= nPoints)

print(f"\nFirst {nPoints} correct points:")
prj.print_multipoles(
    l_values= lValues,
    s_array= sMean_array_correct,
    xi_multipoles= multipolesMean_correct,
    n_values= nPoints
)

prj.plot_multipoles_measVScorr(
    l_values= lValues,
    s_unique= s_unique,
    multipoles_measured= multipolesMean_measured,
    multipoles_correct= multipolesMean_correct,
    err_multipoles_measured=multipolesStd_measured,
    err_multipoles_correct=multipolesStd_correct,
    base_path= "graphs/z1")

prj.plot_multipoles_ratio(
    l_values= lValues,
    s_unique= s_unique,
    multipoles_ratio= multipolesMean_ratio,
    err_multipoles_ratio= multipolesStd_ratio,
    base_path= "graphs/z1",
    ylim= (0,2)
)


plt.show()
plt.close('all')