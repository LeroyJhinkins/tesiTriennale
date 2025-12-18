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
s_matrix_meas, mu_matrix_meas, dd_matrix_meas, dr_matrix_meas, rr_matrix_meas, Ndd_matrix_meas, Ndr_matrix_meas, Nrr_matrix_meas = \
    read.readFITS_pairs_series_SMU(
        "data/z1_data/z1_measured",
        nFiles,
        nElements
    )

print(f"Reading {nFiles} correct files...")
s_matrix_corr, mu_matrix_corr, dd_matrix_corr, dr_matrix_corr, rr_matrix_corr, Ndd_matrix_corr, Ndr_matrix_corr, Nrr_matrix_corr = \
    read.readFITS_pairs_series_SMU(
        "data/z1_data/z1_correct",
        nFiles,
        nElements
    )


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
        base_path= "graphs/z1_rebin",
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
        base_path= "graphs/z1_rebin",
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
    base_path= "graphs/z1_rebin",
    v_min= -1.5,
    v_max= 1.5,
    z_max= 2
)

bm.plot_imshow_ratio(
    coords= "SMU",
    x_array= muMean_array_meas_reb,
    y_array= sMean_array_meas_reb,
    z_array_ratio= xiMean_array_ratio_reb,
    base_path= "graphs/z1_rebin",
    v_min= -1.5,
    v_max= 1.5,
    interp="nearest",
    z_max= 2
)


# custering wedges ----------------------------------------------------------------------------------------------------------------
# clustering wedge (s) = (int_(mu_min)^(mu_max) xi(s,mu) dmu) / (mu_max - mu_min)
# for custering wedges we often use 0 < mu < 1
# however we have -1 < mu < 1
# but for galaxies pairs the correlation function should be symmetrical xi(s,mu) = xi(s,-mu) --> I verified it's true

nWedges = 2
print(f"\n===== Calculating {nWedges} clustering wedges =====")

# we need to compute clustering wedges but we dont do wedge(mean) we do mean(wedges) to have an accurate estimate of the wedges
# not rebinned ones
# wedges_measured_all = []
# wedges_correct_all = []
# wedges_ratio_all = []
# for i in range(nFiles):
#     wedges_i_measured = prj.compute_clusteringWedges(
#         nWedges,
#         s_matrix_meas[i],
#         mu_matrix_meas[i],
#         xi_matrix_meas[i]
#     )
#     wedges_measured_all.append(wedges_i_measured)

#     wedges_i_correct = prj.compute_clusteringWedges(
#         nWedges,
#         s_matrix_corr[i],
#         mu_matrix_corr[i],
#         xi_matrix_corr[i]
#     )
#     wedges_correct_all.append(wedges_i_correct)

#     wedges_i_ratio = wedges_i_measured / wedges_i_correct
#     wedges_ratio_all.append(wedges_i_ratio)

# wedges_measured_all = np.array(wedges_measured_all) # shape = (1000, nWedges, n_s_unique)
# wedgesMean_measured = wedges_measured_all.mean(axis=0)

# wedges_correct_all = np.array(wedges_correct_all) # shape = (1000, nWedges, n_s_unique)
# wedgesMean_correct = wedges_correct_all.mean(axis=0)

# wedges_ratio_all = np.array(wedges_ratio_all) # shape = (1000, nWedges, n_s_unique)
# wedgesMean_ratio = wedges_ratio_all.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
                                                   # and we want to be the most accurate possible => we need mean(ratio)

# rebinned ones
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
wedgesStd_measured_reb = wedges_measured_all_reb.std(axis=0, ddof=1)

wedges_correct_all_reb = np.array(wedges_correct_all_reb) # shape = (1000, nWedges, n_s_unique)
wedgesMean_correct_reb = wedges_correct_all_reb.mean(axis=0)
wedgesStd_correct_reb = wedges_correct_all_reb.std(axis=0, ddof=1)

wedges_ratio_all_reb = np.array(wedges_ratio_all_reb) # shape = (1000, nWedges, n_s_unique)
wedgesMean_ratio_reb = wedges_ratio_all_reb.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
wedgesStd_ratio_reb = wedges_ratio_all_reb.std(axis=0, ddof=1) # and we want to be the most accurate possible => we need mean(ratio)

# covariance matrices
wedgesCov_measured_reb = []
wedgesCov_correct_reb = []
for w in range(nWedges):
    cov_w_measured_reb = np.cov(wedges_measured_all_reb[:,w,:], rowvar=False)
    wedgesCov_measured_reb.append(cov_w_measured_reb)
    
    cov_w_correct_reb = np.cov(wedges_correct_all_reb[:,w,:], rowvar=False)
    wedgesCov_correct_reb.append(cov_w_correct_reb)

wedgesCov_measured_reb = np.array(wedgesCov_measured_reb) # shape = (nWedges, n_s_unique, n_s_unique)
wedgesCov_correct_reb = np.array(wedgesCov_correct_reb) # shape = (nWedges, n_s_unique, n_s_unique)

mu_edges = np.linspace(0.0, 1.0, nWedges + 1)
for w in range(nWedges):
    mu_min = mu_edges[w]
    mu_max = mu_edges[w + 1]
    label = fr"Correlation Matrix for $\xi_{{[{mu_min:.2f}, \,{mu_max:.2f}]}}$"

    bm.plot_correlationMatrix(
        coords= "SMU",
        scale= s_unique_reb,
        cov= wedgesCov_measured_reb[w],
        base_path= "graphs/z1_rebin",
        kind= "measured",
        fig= label,
        filename= f"wedgeCov_[{mu_min:.2f},{mu_max:.2f}]_meas"
    )
    
    bm.plot_correlationMatrix(
        coords= "SMU",
        scale= s_unique_reb,
        cov= wedgesCov_correct_reb[w],
        base_path= "graphs/z1_rebin",
        kind= "correct",
        fig= label,
        filename= f"wedgeCov_[{mu_min:.2f},{mu_max:.2f}]_corr"
    )

# plotting
prj.plot_clusteringWedges_measVScorr(
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges_measured= wedgesMean_measured_reb,
    wedges_correct= wedgesMean_correct_reb,
    err_wedges_measured= wedgesStd_measured_reb,
    err_wedges_correct= wedgesStd_correct_reb,
    base_path= "graphs/z1_rebin"
)

prj.plot_clusteringWedges_ratio(
    n_wedges= nWedges,
    s_unique= s_unique_reb,
    wedges_ratio= wedgesMean_ratio_reb,
    err_wedges_ratio= wedgesStd_ratio_reb,
    base_path= "graphs/z1_rebin",
    ylim=(0,2)
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
    base_path= "graphs/z1_rebin"
)

prj.compute_muMax_ratio(
    s_matrix= s_matrix_meas_reb,
    mu_matrix= mu_matrix_meas_reb,
    xi_matrix_measured= xi_matrix_meas_reb,
    xi_matrix_correct= xi_matrix_corr_reb,
    mu_max_values= np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
    base_path= "graphs/z1_rebin",
    ylim=(0.6,0.8)
)


# multipoles projection -----------------------------------------------------------------------------------------------------------
# same thing for the multipoles
lValues = np.array([0])

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
multipolesStd_measured_reb = multipoles_measured_all_reb.std(axis=0, ddof=1)

multipoles_correct_all_reb = np.array(multipoles_correct_all_reb) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_correct_reb = multipoles_correct_all_reb.mean(axis=0)
multipolesStd_correct_reb = multipoles_correct_all_reb.std(axis=0, ddof=1)

multipoles_ratio_all_reb = np.array(multipoles_ratio_all_reb) # shape = (1000, n_s_unique, n_l_values)
multipolesMean_ratio_reb = multipoles_ratio_all_reb.mean(axis=0) # we did it like this because mean(ratio) != ratio(mean)
multipolesStd_ratio_reb = multipoles_ratio_all_reb.std(axis=0, ddof=1) # and we want to be the most accurate possible => we need mean(ratio) 

# covariance matrices
multipolesCov_measured_reb = []
multipolesCov_correct_reb = []
for l in range(len(lValues)):
    cov_l_measured_reb = np.cov(multipoles_measured_all_reb[:,:,l], rowvar=False)
    multipolesCov_measured_reb.append(cov_l_measured_reb)
    
    cov_l_correct_reb = np.cov(multipoles_correct_all_reb[:,:,l], rowvar=False)
    multipolesCov_correct_reb.append(cov_l_correct_reb)

multipolesCov_measured_reb = np.array(multipolesCov_measured_reb) # shape = (n_l_values, n_s_unique, n_s_unique)
multipolesCov_correct_reb = np.array(multipolesCov_correct_reb) # shape = (n_l_values, n_s_unique, n_s_unique)

for i, l in enumerate(lValues):
    label = fr"Correlation Matrix for $\xi_{{{l}}}$"

    bm.plot_correlationMatrix(
        coords= "SMU",
        scale= s_unique_reb,
        cov= multipolesCov_measured_reb[i],
        base_path= "graphs/z1_rebin",
        kind= "measured",
        fig= label,
        filename= f"multiCov_{l}_meas"
    )
    
    bm.plot_correlationMatrix(
        coords= "SMU",
        scale= s_unique_reb,
        cov= multipolesCov_correct_reb[i],
        base_path= "graphs/z1_rebin",
        kind= "correct",
        fig= label,
        filename= f"multiCov_{l}_corr"
    )

# plotting
nPoints = 5
print("\n===== Calculating monopoles =====")
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

prj.plot_multipole_measVScorr(
    l_value= lValues[0],
    s_unique= s_unique_reb,
    multipole_measured= multipolesMean_measured_reb[:,0],
    multipole_correct= multipolesMean_correct_reb[:,0],
    err_multipole_measured=multipolesStd_measured_reb[:,0],
    err_multipole_correct=multipolesStd_correct_reb[:,0],
    base_path= "graphs/z1_rebin"
)

prj.plot_multipole_ratio(
    l_value= lValues[0],
    s_unique= s_unique_reb,
    multipole_ratio= multipolesMean_ratio_reb[:,0],
    err_multipole_ratio= multipolesStd_ratio_reb[:,0],
    base_path= "graphs/z1_rebin",
    ylim= (0,2)
)


plt.show()
plt.close('all')