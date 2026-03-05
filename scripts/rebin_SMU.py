import sys
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


# redshift bin to read ------------------------------------------------------------------------------------------------------------
redshift_bin = int(sys.argv[1])
valid_bins = [1,2,3,4]
if redshift_bin not in valid_bins:
    raise ValueError(f"Please declare redshift bin to read, must be one of {valid_bins}, got {redshift_bin}")

f_array = [0.16, 0.23, 0.25, 0.14] # found in Ilaria's paper
f = f_array[redshift_bin - 1] # interlopers fraction for the given redshift bin


# read FITS files -----------------------------------------------------------------------------------------------------------------
nFiles = 1000
nElements = 40000

print(f"\nReading {nFiles} measured files...")
s_matrix_meas, mu_matrix_meas, dd_matrix_meas, dr_matrix_meas, rr_matrix_meas, Ndd_matrix_meas, Ndr_matrix_meas, Nrr_matrix_meas = \
    read.readFITS_pairs_series_SMU(
        f"data/z{redshift_bin}_data/z{redshift_bin}_measured",
        nFiles,
        nElements,
        "measured",
        redshift_bin
    )

print(f"Reading {nFiles} correct files...")
s_matrix_corr, mu_matrix_corr, dd_matrix_corr, dr_matrix_corr, rr_matrix_corr, Ndd_matrix_corr, Ndr_matrix_corr, Nrr_matrix_corr = \
    read.readFITS_pairs_series_SMU(
        f"data/z{redshift_bin}_data/z{redshift_bin}_correct",
        nFiles,
        nElements,
        "correct",
        redshift_bin
    )


# this is for the errors: std computes the error of the single measure,
# but, we when we do more realisations, we actually need to compute the error of the mean = error / sqrt(nFiles)
sigmaMean = True
if sigmaMean:
    normaliz = 1.0 / np.sqrt(nFiles)
else:
    normaliz = 1.0


# rebinning and compute 2PCF ------------------------------------------------------------------------------------------------------
# we want to reduce noise and we don't need to have the fine binning we have (200 bins in s)
# so we rebin s into 40 bins (5 Mpc/h per bin), while maintaining mu-bins unchanged (200)
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


s_matrix_meas_reb, mu_matrix_meas_reb, dd_matrix_meas_reb, dr_matrix_meas_reb, rr_matrix_meas_reb = \
    rebin.rebin_SMU(
        s_matrix= s_matrix_meas,
        mu_matrix= mu_matrix_meas,
        dd_matrix= dd_matrix_meas,
        dr_matrix= dr_matrix_meas,
        rr_matrix= rr_matrix_meas,
        delta_s= 5
    )

xi_matrix_meas_reb = rebin.compute_xi(
    dd= dd_matrix_meas_reb,
    dr= dr_matrix_meas_reb,
    rr= rr_matrix_meas_reb,
    Ndd= Ndd_matrix_meas,
    Ndr= Ndr_matrix_meas,
    Nrr= Nrr_matrix_meas
)

s_matrix_corr_reb, mu_matrix_corr_reb, dd_matrix_corr_reb, dr_matrix_corr_reb, rr_matrix_corr_reb = \
    rebin.rebin_SMU(
        s_matrix= s_matrix_corr,
        mu_matrix= mu_matrix_corr,
        dd_matrix= dd_matrix_corr,
        dr_matrix= dr_matrix_corr,
        rr_matrix= rr_matrix_corr,
        delta_s= 5
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
sMean_array_meas = np.mean(s_matrix_meas, axis=0) # this is probably useless
muMean_array_meas = np.mean(mu_matrix_meas, axis=0) # just a check
xiMean_array_meas = np.mean(xi_matrix_meas, axis=0)

sMean_array_corr = np.mean(s_matrix_corr, axis=0) # this is probably useless
muMean_array_corr = np.mean(mu_matrix_corr, axis=0) # just a check
xiMean_array_corr = np.mean(xi_matrix_corr, axis=0)

xiMean_array_ratio = np.mean(xi_matrix_ratio, axis=0)

# rebinned ones
sMean_array_meas_reb = np.mean(s_matrix_meas_reb, axis=0) # this is probably useless
muMean_array_meas_reb = np.mean(mu_matrix_meas_reb, axis=0) # just a check
xiMean_array_meas_reb = np.mean(xi_matrix_meas_reb, axis=0)

sMean_array_corr_reb = np.mean(s_matrix_corr_reb, axis=0) # this is probably useless
muMean_array_corr_reb = np.mean(mu_matrix_corr_reb, axis=0) # just a check
xiMean_array_corr_reb = np.mean(xi_matrix_corr_reb, axis=0)

xiMean_array_ratio_reb = np.mean(xi_matrix_ratio_reb, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(muMean_array_meas_reb, muMean_array_corr_reb, atol=1e-4) and \
    np.allclose(sMean_array_meas_reb, sMean_array_corr_reb, atol=1e-4): # this should always be true

    vmin = min(xiMean_array_meas_reb.min(), xiMean_array_corr_reb.min())
    vmax = max(xiMean_array_meas_reb.max(), xiMean_array_corr_reb.max())

    # plot measured
    MU, S, _ = bm.plot_contourf(
        coords= "SMU",
        x_array= muMean_array_meas_reb,
        y_array= sMean_array_meas_reb,
        z_array= xiMean_array_meas_reb,
        base_path= f"graphs/z{redshift_bin}_rebin",
        kind= "measured",
        v_min= vmin,
        v_max= vmax
    )

    # plot correct
    bm.plot_contourf(
        coords= "SMU",
        x_array= muMean_array_corr_reb,
        y_array= sMean_array_corr_reb,
        z_array= xiMean_array_corr_reb,
        base_path= f"graphs/z{redshift_bin}_rebin",
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
    coords= "SMU",
    x_array= muMean_array_meas_reb,
    y_array= sMean_array_meas_reb,
    z_array_ratio= xiMean_array_ratio_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    v_min= -1.5,
    v_max= 1.5,
    z_max= 2
)

bm.plot_imshow_ratio(
    coords= "SMU",
    x_array= muMean_array_meas_reb,
    y_array= sMean_array_meas_reb,
    z_array_ratio= xiMean_array_ratio_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    v_min= -1.5,
    v_max= 1.5,
    interp="nearest",
    z_max= 2
)

plt.show()
plt.close('all')


# custering wedges ----------------------------------------------------------------------------------------------------------------
# clustering wedge (s) = (int_(mu_min)^(mu_max) xi(s,mu) dmu) / (mu_max - mu_min)
# for custering wedges we often use 0 < mu < 1
# however we have -1 < mu < 1
# but for galaxies pairs the correlation function should be symmetrical xi(s,mu) = xi(s,-mu) --> I verified it's true

nWedges = 2
print(f"\n===== Calculating {nWedges} clustering wedges =====")

# we need to compute clustering wedges but we dont do wedge(mean) we do mean(wedges) to have an accurate estimate of the wedges
s_unique = np.unique(sMean_array_corr)
s_unique_reb = np.unique(sMean_array_corr_reb)

wedges_measured_all_reb = []
wedges_correct_all_reb = []
wedges_ratio_all_reb = []
for i in range(nFiles):
    wedges_i_measured_reb = prj.compute_clusteringWedges(
        nWedges,
        s_matrix_meas_reb[i],
        mu_matrix_meas_reb[i],
        xi_matrix_meas_reb[i]
    )
    wedges_measured_all_reb.append(wedges_i_measured_reb)

    wedges_i_correct_reb = prj.compute_clusteringWedges(
        nWedges,
        s_matrix_corr_reb[i],
        mu_matrix_corr_reb[i],
        xi_matrix_corr_reb[i]
    )
    wedges_correct_all_reb.append(wedges_i_correct_reb)

    wedges_i_ratio_reb = wedges_i_measured_reb / wedges_i_correct_reb
    wedges_ratio_all_reb.append(wedges_i_ratio_reb)

wedges_measured_all_reb = np.array(wedges_measured_all_reb) # shape = (1000, nWedges, n_s_unique)
wedgesMean_measured_reb = wedges_measured_all_reb.mean(axis=0)
wedgesStd_measured_reb = wedges_measured_all_reb.std(axis=0, ddof=1) * normaliz

wedges_correct_all_reb = np.array(wedges_correct_all_reb) # shape = (1000, nWedges, n_s_unique)
wedgesMean_correct_reb = wedges_correct_all_reb.mean(axis=0)
wedgesStd_correct_reb = wedges_correct_all_reb.std(axis=0, ddof=1) * normaliz

wedges_ratio_all_reb = np.array(wedges_ratio_all_reb) # shape = (1000, nWedges, n_s_unique)
wedgesMean_ratio_reb = wedges_ratio_all_reb.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
wedgesStd_ratio_reb = wedges_ratio_all_reb.std(axis=0, ddof=1) * normaliz # and we want to be the most accurate possible => we need mean(ratio)

# covariance matrices
# reorder to (nFiles, nWedges * n_s_unique) in order to get the full covariance matrix
n_s_unique = len(s_unique_reb)
wedges_measured_all_reb_flatten = wedges_measured_all_reb.reshape(nFiles, nWedges * n_s_unique)
wedges_correct_all_reb_flatten = wedges_correct_all_reb.reshape(nFiles, nWedges * n_s_unique)

wedgesCov_measured_reb = np.cov(wedges_measured_all_reb_flatten, rowvar=False) # (nWedges * n_s_unique, nWedges * n_s_unique)
wedgesCov_correct_reb = np.cov(wedges_correct_all_reb_flatten, rowvar=False) # (nWedges * n_s_unique, nWedges * n_s_unique)

s_wedges_reb = np.tile(s_unique_reb, nWedges)

label = r"Correlation Matrix for $\xi_{[\mu_\mathrm{min}, \mu_\mathrm{max}]}$"

bm.plot_correlationMatrix(
    coords= "SMU",
    scale= s_wedges_reb,
    cov= wedgesCov_measured_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    kind= "measured",
    fig= label,
    filename= f"wedgesCov_meas",
    n_wedges= nWedges
)

bm.plot_correlationMatrix(
    coords= "SMU",
    scale= s_wedges_reb,
    cov= wedgesCov_correct_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    kind= "correct",
    fig= label,
    filename= f"wedgesCov_corr",
    n_wedges= nWedges
)

# plotting
prj.plot_clusteringWedges_measVScorr(
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges_measured= wedgesMean_measured_reb,
    wedges_correct= wedgesMean_correct_reb,
    err_wedges_measured= wedgesStd_measured_reb,
    err_wedges_correct= wedgesStd_correct_reb,
    base_path= f"graphs/z{redshift_bin}_rebin"
)

prj.plot_clusteringWedges_ratio(
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges_ratio= wedgesMean_ratio_reb,
    err_wedges_ratio= wedgesStd_ratio_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    ylim=(0.25,1.25),
    yref= (1-f)**2
)

# BAO peak finding
s_peak_array_measured_reb, xi_peak_array_measured_reb, s_low_measured_reb, s_high_measured_reb = prj.compute_clusteringWedges_BAOpeaks( # type: ignore
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges= wedgesMean_measured_reb,
    err_wedges= wedgesStd_measured_reb,
    s_min= 90,
    s_max= 110
)

print("Printing measured BAO peak...")
prj.print_clusteringWedges_BAOintervals(
    n_wedges= nWedges,
    s_peak= s_peak_array_measured_reb,
    xi_peak= xi_peak_array_measured_reb,
    s_low= s_low_measured_reb,
    s_high= s_high_measured_reb)

s_peak_array_correct_reb, xi_peak_array_correct_reb, s_low_correct_reb, s_high_correct_reb = prj.compute_clusteringWedges_BAOpeaks( # type: ignore
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges= wedgesMean_correct_reb,
    err_wedges= wedgesStd_correct_reb,
    s_min= 90,
    s_max= 110
)

print("Printing correct BAO peak...")
prj.print_clusteringWedges_BAOintervals(
    n_wedges= nWedges,
    s_peak= s_peak_array_correct_reb,
    xi_peak= xi_peak_array_correct_reb,
    s_low= s_low_correct_reb,
    s_high= s_high_correct_reb
)

# we integrate xi(mu, s) from mu=0 to mu=mu_max
# in order to find the mu_max corresponding to the beginning of the scale dependance in s
prj.compute_muMax(
    s_matrix= s_matrix_meas_reb,
    mu_matrix= mu_matrix_meas_reb,
    xi_matrix= xi_matrix_meas_reb,
    mu_max_values= np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    base_path= f"graphs/z{redshift_bin}_rebin"
)

prj.compute_muMax_ratio(
    s_matrix= s_matrix_meas_reb,
    mu_matrix= mu_matrix_meas_reb,
    xi_matrix_measured= xi_matrix_meas_reb,
    xi_matrix_correct= xi_matrix_corr_reb,
    mu_max_values= np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    base_path= f"graphs/z{redshift_bin}_rebin",
    ylim=(0.6,0.8)
)

plt.show()
plt.close('all')


# multipoles projection -----------------------------------------------------------------------------------------------------------
# same thing for the multipoles
lValues = np.array([0,2,4])
print("\n===== Calculating monopoles =====")

# not rebinned ones
multipoles_measured_all = []
for i in range(nFiles):
    multipoles_i_measured = prj.compute_multipoles(
        lValues,
        s_matrix_meas[i],
        mu_matrix_meas[i],
        xi_matrix_meas[i]
    )
    multipoles_measured_all.append(multipoles_i_measured)

multipoles_measured_all = np.array(multipoles_measured_all) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_measured = multipoles_measured_all.mean(axis=0)
multipolesStd_measured = multipoles_measured_all.std(axis=0, ddof=1) * normaliz

# rebinned ones
multipoles_measured_all_reb = []
multipoles_correct_all_reb = []
multipoles_ratio_all_reb = []
for i in range(nFiles):
    multipoles_i_measured_reb = prj.compute_multipoles(
        lValues,
        s_matrix_meas_reb[i],
        mu_matrix_meas_reb[i],
        xi_matrix_meas_reb[i]
    )
    multipoles_measured_all_reb.append(multipoles_i_measured_reb)

    multipoles_i_correct_reb = prj.compute_multipoles(
        lValues,
        s_matrix_corr_reb[i],
        mu_matrix_corr_reb[i],
        xi_matrix_corr_reb[i]
    )
    multipoles_correct_all_reb.append(multipoles_i_correct_reb)

    multipoles_i_ratio_reb = multipoles_i_measured_reb / multipoles_i_correct_reb
    multipoles_ratio_all_reb.append(multipoles_i_ratio_reb)

multipoles_measured_all_reb = np.array(multipoles_measured_all_reb) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_measured_reb = multipoles_measured_all_reb.mean(axis=0)
multipolesStd_measured_reb = multipoles_measured_all_reb.std(axis=0, ddof=1) * normaliz

multipoles_correct_all_reb = np.array(multipoles_correct_all_reb) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_correct_reb = multipoles_correct_all_reb.mean(axis=0)
multipolesStd_correct_reb = multipoles_correct_all_reb.std(axis=0, ddof=1) * normaliz

multipoles_ratio_all_reb = np.array(multipoles_ratio_all_reb) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_ratio_reb = multipoles_ratio_all_reb.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
multipolesStd_ratio_reb = multipoles_ratio_all_reb.std(axis=0, ddof=1) * normaliz # and we want to be the most accurate possible => we need mean(ratio) 

# covariance matrices
# reorder to (nFiles, n_l_values * n_s_unique) in order to get the full covariance matrix
multipoles_measured_all_reb_flatten = np.transpose(multipoles_measured_all_reb, (0, 2, 1)).reshape(nFiles, len(lValues) * n_s_unique)
multipoles_correct_all_reb_flatten = np.transpose(multipoles_correct_all_reb, (0, 2, 1)).reshape(nFiles, len(lValues) * n_s_unique)

multipolesCov_measured_reb = np.cov(multipoles_measured_all_reb_flatten, rowvar=False) # (n_l_values * n_s_unique, n_l_values * n_s_unique)
multipolesCov_correct_reb = np.cov(multipoles_correct_all_reb_flatten, rowvar=False) # (n_l_values * n_s_unique, n_l_values * n_s_unique)

s_multipoles_reb = np.tile(s_unique_reb, len(lValues))

label = r"Correlation Matrix for $\xi_\ell$"

bm.plot_correlationMatrix(
    coords= "SMU",
    scale= s_multipoles_reb,
    cov= multipolesCov_measured_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    kind= "measured",
    fig= label,
    filename= f"multiCov_meas",
    l_values=lValues
)

bm.plot_correlationMatrix(
    coords= "SMU",
    scale= s_multipoles_reb,
    cov= multipolesCov_correct_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    kind= "correct",
    fig= label,
    filename= f"multiCov_corr",
    l_values=lValues
)

# plotting
nPoints = 5
print(f"First {nPoints} measured points:")
prj.print_multipoles(
    l_values= lValues,
    s_array= sMean_array_meas_reb,
    xi_multipoles= multipolesMean_measured_reb,
    n_values= nPoints
)

print(f"\nFirst {nPoints} correct points:")
prj.print_multipoles(
    l_values= lValues,
    s_array= sMean_array_corr_reb,
    xi_multipoles= multipolesMean_correct_reb,
    n_values= nPoints
)

prj.plot_multipoles_rebinVSnot(
    l_values= lValues,
    s_unique= s_unique,
    s_unique_rebin= s_unique_reb,
    multipoles= multipolesMean_measured,
    multipoles_rebin= multipolesMean_measured_reb,
    err_multipoles= multipolesStd_measured,
    err_multipoles_rebin= multipolesStd_measured_reb,
    base_path= f"graphs/z{redshift_bin}_rebin"
)

prj.plot_multipoles_measVScorr(
    l_values= lValues,
    s_unique= s_unique_reb,
    multipoles_measured= multipolesMean_measured_reb,
    multipoles_correct= multipolesMean_correct_reb,
    err_multipoles_measured= multipolesStd_measured_reb,
    err_multipoles_correct= multipolesStd_correct_reb,
    base_path= f"graphs/z{redshift_bin}_rebin"
)

prj.plot_multipoles_ratio(
    l_values= np.array([0,2]),
    s_unique= s_unique_reb,
    multipoles_ratio= multipolesMean_ratio_reb,
    err_multipoles_ratio= multipolesStd_ratio_reb,
    base_path= f"graphs/z{redshift_bin}_rebin",
    ylim= (0.25,1.25),
    yref= (1-f)**2
)


# save data for model fitting
np.save(f"outData/z{redshift_bin}_rebin/s_unique_reb.npy", s_unique_reb)

np.save(f"outData/z{redshift_bin}_rebin/mean_wedges_measured.npy", wedgesMean_measured_reb[0])
np.save(f"outData/z{redshift_bin}_rebin/cov_wedges_measured.npy", wedgesCov_measured_reb[0:n_s_unique, 0:n_s_unique])
np.save(f"outData/z{redshift_bin}_rebin/mean_wedges_correct.npy", wedgesMean_correct_reb[0])
np.save(f"outData/z{redshift_bin}_rebin/cov_wedges_correct.npy", wedgesCov_correct_reb[0:n_s_unique, 0:n_s_unique])

np.save(f"outData/z{redshift_bin}_rebin/mean_multipoles_measured.npy", multipoles_measured_all_reb_flatten.mean(axis=0))
np.save(f"outData/z{redshift_bin}_rebin/cov_multipoles_measured.npy", multipolesCov_measured_reb)
np.save(f"outData/z{redshift_bin}_rebin/mean_multipoles_correct.npy", multipoles_correct_all_reb_flatten.mean(axis=0))
np.save(f"outData/z{redshift_bin}_rebin/cov_multipoles_correct.npy", multipolesCov_correct_reb)


plt.show()
plt.close('all')