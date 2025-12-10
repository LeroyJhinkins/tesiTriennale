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
nFiles = 1000
nElements = 40000

print(f"\nReading {nFiles} measured files...")
s_matrix_meas, mu_matrix_meas, dd_matrix_meas, dr_matrix_meas, rr_matrix_meas = read.readFITS_pairs_series_SMU("data/z1_data/z1_measured",
                                                                                                               nFiles, nElements)

print(f"Reading {nFiles} correct files...")
s_matrix_corr, mu_matrix_corr, dd_matrix_corr, dr_matrix_corr, rr_matrix_corr = read.readFITS_pairs_series_SMU("data/z1_data/z1_correct",
                                                                                                               nFiles, nElements)


# rebinning and compute 2PCF ------------------------------------------------------------------------------------------------------
# we want to reduce noise and we don't need to have the fine binning we have (200 bins in s)
# so we rebin s into 40 bins (5 Mpc/h per bin), while maintaining mu-bins unchanged (200)
sReb_matrix_meas, muReb_matrix_meas, ddReb_matrix_meas, drReb_matrix_meas, rrReb_matrix_meas = rebin.rebin_SMU(s_matrix= s_matrix_meas,
                                                                                                               mu_matrix= mu_matrix_meas,
                                                                                                               dd_matrix= dd_matrix_meas,
                                                                                                               dr_matrix= dr_matrix_meas,
                                                                                                               rr_matrix= rr_matrix_meas,
                                                                                                               delta_s= 5)

xiReb_matrix_meas = rebin.compute_xi(ddReb_matrix_meas, drReb_matrix_meas, rrReb_matrix_meas)


sReb_matrix_corr, muReb_matrix_corr, ddReb_matrix_corr, drReb_matrix_corr, rrReb_matrix_corr = rebin.rebin_SMU(s_matrix= s_matrix_corr,
                                                                                                               mu_matrix= mu_matrix_corr,
                                                                                                               dd_matrix= dd_matrix_corr,
                                                                                                               dr_matrix= dr_matrix_corr,
                                                                                                               rr_matrix= rr_matrix_corr,
                                                                                                               delta_s= 5)

xiReb_matrix_corr = rebin.compute_xi(ddReb_matrix_corr, drReb_matrix_corr, rrReb_matrix_corr)


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
sRebMean_array_meas = np.mean(sReb_matrix_meas, axis=0) # this is probably useless
muRebMean_array_meas = np.mean(muReb_matrix_meas, axis=0) # just a check
xiRebMean_array_meas = np.mean(xiReb_matrix_meas, axis=0)
# xiStd_array_meas = np.std(xi_matrix_meas, axis=0)

sRebMean_array_corr = np.mean(sReb_matrix_corr, axis=0) # this is probably useless
muRebMean_array_corr = np.mean(muReb_matrix_corr, axis=0) # just a check
xiRebMean_array_corr = np.mean(xiReb_matrix_corr, axis=0)
# xiStd_array_corr = np.std(xi_matrix_corr, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(muRebMean_array_meas, muRebMean_array_corr, atol=1e-4) and \
    np.allclose(sRebMean_array_meas, sRebMean_array_corr, atol=1e-4): # this should always be true

    # plot measured
    MU, S, _ = bm.plot_contourf(coords= "SMU",
                                x_array= muRebMean_array_meas,
                                y_array= sRebMean_array_meas,
                                z_array= xiRebMean_array_meas,
                                base_path= "graphs/z1_rebin",
                                kind= "measured")

    # plot correct
    bm.plot_contourf(coords= "SMU",
                     x_array= muRebMean_array_corr,
                     y_array= sRebMean_array_corr,
                     z_array= xiRebMean_array_corr,
                     base_path= "graphs/z1_rebin",
                     kind="correct")

else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")


# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
# plot in s and mu
xiMean_ratio = bm.plot_contourf_ratio(coords= "SMU",
                                      x_array= muRebMean_array_meas,
                                      y_array= sRebMean_array_meas,
                                      z_array_measured= xiRebMean_array_meas,
                                      z_array_correct= xiRebMean_array_corr,
                                      base_path= "graphs/z1_rebin",
                                      v_min= -1.5,
                                      v_max= 1.5,
                                      z_max= 2)

xiMean_ratio = bm.plot_imshow_ratio(coords= "SMU",
                                    x_array= muRebMean_array_meas,
                                    y_array= sRebMean_array_meas,
                                    z_array_measured= xiRebMean_array_meas,
                                    z_array_correct= xiRebMean_array_corr,
                                    base_path= "graphs/z1_rebin",
                                    v_min= -1.5,
                                    v_max= 1.5,
                                    interp="nearest",
                                    z_max= 2)


# custering wedges ----------------------------------------------------------------------------------------------------------------
# clustering wedge (s) = (int_(mu_min)^(mu_max) xi(s,mu) dmu) / (mu_max - mu_min)
# for custering wedges we often use 0 < mu < 1
# however we have -1 < mu < 1
# but for galaxies pairs the correlation function should be symmetrical xi(s,mu) = xi(s,-mu) --> I verified it's true

nWedges = 2
print(f"\n===== Calculating {nWedges} clustering wedges =====")

s_unique = np.unique(sRebMean_array_corr)

# we need to comute clustering wedges but we dont do wedge(mean) we do mean(wedges) to have an accurate estimate of the wedges
wedges_measured_all = []
for i in range(nFiles):
    wedges_i = prj.compute_clusteringWedges(
        nWedges,
        sReb_matrix_meas[i],
        muReb_matrix_meas[i],
        xiReb_matrix_meas[i]
    )
    wedges_measured_all.append(wedges_i)
wedges_measured_all = np.array(wedges_measured_all) # shape = (1000, nWedges, n_s_unique)

wedgesMean_measured = wedges_measured_all.mean(axis=0)
wedgesStd_measured = wedges_measured_all.std(axis=0)

wedges_correct_all = []
for i in range(nFiles):
    wedges_i = prj.compute_clusteringWedges(
        nWedges,
        sReb_matrix_corr[i],
        muReb_matrix_corr[i],
        xiReb_matrix_corr[i]
    )
    wedges_correct_all.append(wedges_i)
wedges_correct_all = np.array(wedges_correct_all) # shape = (1000, nWedges, n_s_unique)

wedgesMean_correct = wedges_correct_all.mean(axis=0)
wedgesStd_correct = wedges_correct_all.std(axis=0)

prj.plot_clusteringWedges_measVScorr(n_wedges= nWedges,
                                     s_unique= s_unique,
                                     wedges_measured= wedgesMean_measured,
                                     wedges_correct= wedgesMean_correct,
                                     err_wedges_measured= wedgesStd_measured,
                                     err_wedges_correct= wedgesStd_correct,
                                     base_path= "graphs/z1_rebin")

ratio, errRatio = prj.plot_clusteringWedges_ratio(n_wedges= nWedges,
                                                  s_unique= s_unique,
                                                  wedges_measured= wedgesMean_measured,
                                                  wedges_correct= wedgesMean_correct,
                                                  err_wedges_measured= wedgesStd_measured,
                                                  err_wedges_correct= wedgesStd_correct,
                                                  base_path= "graphs/z1_rebin",
                                                  ylim=(0,2))

s_peak_array_measured, xi_peak_array_measured, s_low_measured, s_high_measured = prj.compute_clusteringWedges_BAOpeaks(n_wedges= nWedges, # type: ignore
                                                                                                                       s_unique= s_unique,
                                                                                                                       wedges= wedgesMean_measured,
                                                                                                                       err_wedges= wedgesStd_measured,
                                                                                                                       s_min= 90,
                                                                                                                       s_max= 110)

print("Printing measured BAO peak...")
prj.print_clusteringWedges_BAOintervals(n_wedges= nWedges,
                                        s_peak= s_peak_array_measured,
                                        xi_peak= xi_peak_array_measured,
                                        s_low= s_low_measured,
                                        s_high= s_high_measured)

s_peak_array_correct, xi_peak_array_correct, s_low_correct, s_high_correct = prj.compute_clusteringWedges_BAOpeaks(n_wedges= nWedges, # type: ignore
                                                                                                                   s_unique= s_unique,
                                                                                                                   wedges= wedgesMean_correct,
                                                                                                                   err_wedges= wedgesStd_correct,
                                                                                                                   s_min= 90,
                                                                                                                   s_max= 110)

print("Printing correct BAO peak...")
prj.print_clusteringWedges_BAOintervals(n_wedges= nWedges,
                                        s_peak= s_peak_array_correct,
                                        xi_peak= xi_peak_array_correct,
                                        s_low= s_low_correct,
                                        s_high= s_high_correct)

# we integrate xi(mu, s) from mu=0 to mu=mu_max
# in order to find the mu_max corresponding to the beginning of the scale dependance in s
prj.compute_muMax(sRebMean_array_meas, muRebMean_array_meas, xiRebMean_array_meas, 0.7, "graphs/z1_rebin")


# multipoles projection -----------------------------------------------------------------------------------------------------------
# same thing for the multipoles
lValues = np.array([0])

multipoles_measured_all = []
for i in range(nFiles):
    multipoles_i = prj.compute_multipoles(
        lValues,
        sReb_matrix_meas[i],
        muReb_matrix_meas[i],
        xiReb_matrix_meas[i]
    )
    multipoles_measured_all.append(multipoles_i)
multipoles_measured_all = np.array(multipoles_measured_all) # shape = (1000, n_s_unique, n_l_values)

multipolesMean_measured = multipoles_measured_all.mean(axis=0)
multipolesStd_measured = multipoles_measured_all.std(axis=0)

multipoles_correct_all = []
for i in range(nFiles):
    multipoles_i = prj.compute_multipoles(
        lValues,
        sReb_matrix_corr[i],
        muReb_matrix_corr[i],
        xiReb_matrix_corr[i]
    )
    multipoles_correct_all.append(multipoles_i)
multipoles_correct_all = np.array(multipoles_correct_all) # shape = (1000, n_s_unique, n_l_values)

multipolesMean_correct  = multipoles_correct_all.mean(axis=0)
multipolesStd_correct  = multipoles_correct_all.std(axis=0)

nPoints = 5
print("\n===== Calculating monopoles =====")
print(f"First {nPoints} measured points:")
prj.print_multipoles(l_values= lValues,
                    s_array= sRebMean_array_meas,
                    xi_multipoles= multipolesMean_measured,
                    n_values= nPoints)

print(f"\nFirst {nPoints} correct points:")
prj.print_multipoles(l_values= lValues,
                    s_array= sRebMean_array_corr,
                    xi_multipoles= multipolesMean_correct,
                    n_values= nPoints)

prj.plot_multipole_measVScorr(l_value= lValues[0],
                             s_unique= s_unique,
                             multipole_measured= multipolesMean_measured[:,0],
                             multipole_correct= multipolesMean_correct[:,0],
                             err_multipole_measured=multipolesStd_measured[:,0],
                             err_multipole_correct=multipolesStd_correct[:,0],
                             base_path= "graphs/z1_rebin")

multipoleMean_ratio, multipoleStd_ratio = prj.plot_multipole_ratio(l_value= lValues[0],
                                                                  s_unique= s_unique,
                                                                  multipole_measured= multipolesMean_measured[:,0],
                                                                  multipole_correct= multipolesMean_correct[:,0] ,
                                                                  err_multipole_measured= multipolesStd_measured[:,0],
                                                                  err_multipole_correct= multipolesStd_correct[:,0],
                                                                  base_path= "graphs/z1_rebin",
                                                                  ylim=(-2,2))

plt.show()