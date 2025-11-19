import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
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

print("\nReading measured files...")
s_matrix_measured, mu_matrix_measured, xi_matrix_measured = read.readFITS_auto_series_SMU("data/z1_data/z1_measured", nFiles, nElements)

print("Reading correct files...")
s_matrix_correct, mu_matrix_correct, xi_matrix_correct = read.readFITS_auto_series_SMU("data/z1_data/z1_correct", nFiles, nElements)


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
sMean_array_measured = np.mean(s_matrix_measured, axis=0) # this is probably useless
muMean_array_measured = np.mean(mu_matrix_measured, axis=0) # just a check
xiMean_array_measured = np.mean(xi_matrix_measured, axis=0)
# xiStd_array_measured = np.std(xi_matrix_measured, axis=0)

sMean_array_correct = np.mean(s_matrix_correct, axis=0) # this is probably useless
muMean_array_correct = np.mean(mu_matrix_correct, axis=0) # just a check
xiMean_array_correct = np.mean(xi_matrix_correct, axis=0)
# xiStd_array_correct = np.std(xi_matrix_correct, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(muMean_array_measured, muMean_array_correct, atol=1e-4) and \
    np.allclose(sMean_array_measured, sMean_array_correct, atol=1e-4): # this should always be true

    mu_unique = np.unique(muMean_array_measured)
    s_unique = np.unique(sMean_array_measured)
    
    MU, S = np.meshgrid(mu_unique, s_unique)
    XI_MEASURED = xiMean_array_measured.reshape((len(mu_unique), len(s_unique)))
    XI_CORRECT = xiMean_array_correct.reshape((len(mu_unique), len(s_unique)))

    # plot measured
    plt.figure(figsize=(9,8), num="2D map (mu, s) measured")
    contourSMU_measured = plt.contourf(MU, S, XI_MEASURED, levels=20, cmap='turbo')
    
    cbarSMU_measured = plt.colorbar(contourSMU_measured, label=r'$\xi^\mathrm{measured}(\mu, s)$')
    xi_measured_ticks = np.linspace(np.min(xiMean_array_measured), np.max(xiMean_array_measured), 9)
    cbarSMU_measured.set_ticks(xi_measured_ticks)
    cbarSMU_measured.set_ticklabels([f"{tick:.2f}" for tick in xi_measured_ticks])

    mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 5)
    s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 6)
    plt.xticks(mu_ticks, [f"{tick:.0f}" for tick in mu_ticks])
    plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])

    # plt.xlim(0, 15)
    # plt.ylim(0, 15)

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
    plt.title(r'2D map of $\xi^\mathrm{measured}(\mu, s)$')
    plt.tight_layout()
    plt.savefig("graphs/z1/2DmapSMU_measured.pdf", dpi=600)

    # plot correct
    plt.figure(figsize=(9,8), num="2D map (r_p, pi) correct")
    contourSMU_correct = plt.contourf(MU, S, XI_CORRECT, levels=20, cmap='turbo')
    
    cbarSMU_corret = plt.colorbar(contourSMU_correct, label=r'$\xi^\mathrm{correct}(r_p, \pi)$')
    xi_ticks_correct = np.linspace(np.min(xiMean_array_correct), np.max(xiMean_array_correct), 9)
    cbarSMU_corret.set_ticks(xi_ticks_correct)
    cbarSMU_corret.set_ticklabels([f"{tick:.2f}" for tick in xi_ticks_correct])

    plt.xticks(mu_ticks, [f"{tick:.0f}" for tick in mu_ticks])
    plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])
    
    # plt.xlim(0, 15)
    # plt.ylim(0, 15)

    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
    plt.title(r'2D map of $\xi^\mathrm{correct}(\mu, s)$')
    plt.tight_layout()
    plt.savefig("graphs/z1/2DmapSMU_correct.pdf", dpi=600)

else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")


# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
# plot in s and mu

xi_ratio = xiMean_array_measured / xiMean_array_correct

MU, S = np.meshgrid(mu_unique, s_unique)
XI_RATIO = xi_ratio.reshape(len(s_unique), len(mu_unique))

plt.figure(figsize=(9,8), num="Ratio (mu, s)")
contourMUS = plt.contourf(MU, S, XI_RATIO, levels=20, cmap='turbo')

cbarMUS = plt.colorbar(contourMUS, label=r'$\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}$')
xi_ticks = np.linspace(np.min(xi_ratio), np.max(xi_ratio), 9)
cbarMUS.set_ticks(xi_ticks)
cbarMUS.set_ticklabels([f"{tick:.2f}" for tick in xi_ticks])

s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 6)
mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 5)
plt.xticks(mu_ticks, [f"{tick:.0f}" for tick in mu_ticks])
plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])
    
# plt.xlim(0, np.max(mu_unique))
# plt.ylim(0, 10)

plt.xlabel(r'$\mu$')
plt.ylabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}(\mu,s)$')
plt.savefig("graphs/z1/2DmapSMU_ratio.pdf", dpi=600)


# plot in r_p and pi
RP = S * np.sqrt(1 - MU**2)
PI = S * MU

plt.figure(figsize=(9,8), num="Ratio (r_p, pi)")
contourRpPI = plt.contourf(RP, PI, np.log(np.abs(XI_RATIO)), levels=20, cmap='turbo') # we use log to better observe differences in levels
                                                                                      # abs is for avoiding log of negative numbers

cbarRpPI = plt.colorbar(contourRpPI, label=r'$\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}$')
xi_interpol_ticks = np.linspace(np.min(np.log(np.abs(xi_ratio))), np.max(np.log(np.abs(xi_ratio))), 9)
cbarRpPI.set_ticks(xi_interpol_ticks)
cbarRpPI.set_ticklabels([f"{tick:.3f}" for tick in xi_interpol_ticks])

rp_ticks = np.linspace(np.min(RP), np.max(RP), 10)
pi_ticks = np.linspace(np.min(PI), np.max(PI), 10)
plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

# plt.xlim(0,30)
# plt.ylim(-15,15)

plt.xlabel(r'$r_p \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\frac{\xi_\mathrm{measured}}{\xi_\mathrm{correct}}(r_p, \pi)$')
plt.tight_layout()
plt.savefig("graphs/z1/2DmapRpPI_ratio.pdf", dpi=600)


# custering wedges ----------------------------------------------------------------------------------------------------------------
# clustering wedge (s) = (int_(mu_min)^(mu_max) xi(s,mu) dmu) / (mu_max - mu_min)
# for custering wedges we often use 0 < mu < 1
# however we have -1 < mu < 1
# but for galaxies pairs the correlation function should be symmetrical xi(s,mu) = xi(s,-mu) --> I verified it's true

def clusteringWedges(s_array: np.ndarray, mu_array: np.ndarray, xi_array: np.ndarray):

    s_unique = np.unique(s_array)
    xi_perpendicular = np.zeros(len(s_unique)) # 0 < mu < 0.5
    xi_parallel = np.zeros(len(s_unique)) # 0.5 < mu < 1

    for i, s in enumerate(s_unique):
            
        delta_mu = np.float64(0.01)

        mask = (s_array == s) & (mu_array >= 0)
        mu_vals = mu_array[mask]
        xi_vals = xi_array[mask]

        mask_perp = (mu_vals >= 0.0) & (mu_vals < 0.5)
        mask_par  = (mu_vals >= 0.5) & (mu_vals <= 1.0)

        xi_perpendicular[i] = np.sum(xi_vals[mask_perp]) * delta_mu * 2 # 1/0.5 = 2
        xi_parallel[i] = np.sum(xi_vals[mask_par]) * delta_mu * 2 # 1/0.5 = 2
    
    return xi_perpendicular, xi_parallel

xi_perpendicular_measured, xi_parallel_measured = clusteringWedges(sMean_array_measured, muMean_array_measured, xiMean_array_measured)
xi_perpendicular_correct, xi_parallel_correct = clusteringWedges(sMean_array_correct, muMean_array_correct, xiMean_array_correct)

ratio_perpendicular = xi_perpendicular_measured / xi_perpendicular_correct
ratio_parallel = xi_parallel_measured / xi_parallel_correct

s_unique = np.unique(sMean_array_correct)

plt.figure(figsize=(6,6), num="Perpendicular")
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_measured, label=r'$\xi_\perp^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='o', markersize=2, color='blue')
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_correct, label=r'$\xi_\perp^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='s', markersize=2, color='green')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$s^2 \xi_\perp$')
plt.title('Perpendicular wedge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/z1/clustWedge_perp.pdf", dpi=600)


plt.figure(figsize=(6,6), num="Parallel")
plt.plot(s_unique, (s_unique**2) * xi_parallel_measured, label=r'$\xi_\parallel^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='o', markersize=2, color='orange')
plt.plot(s_unique, (s_unique**2) * xi_parallel_correct, label=r'$\xi_\parallel^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='s', markersize=2, color='red')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$s^2 \xi_\parallel$')
plt.title('Parallel wedge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/z1/clustWedge_paral.pdf", dpi=600)


plt.figure(figsize=(8,8), num="Perpendicular vs Parallel")
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_measured, label=r'$\xi_\perp^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='o', markersize=2, color='blue')
plt.plot(s_unique, (s_unique**2) * xi_parallel_measured, label=r'$\xi_\parallel^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='s', markersize=2, color='orange')
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_correct, label=r'$\xi_\perp^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='p', markersize=2, color='green')
plt.plot(s_unique, (s_unique**2) * xi_parallel_correct, label=r'$\xi_\parallel^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='H', markersize=2, color='red')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$s^2 \xi$')
plt.title('Clustering Wedges')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/z1/clustWedge_perpVSparal.pdf", dpi=600)


plt.figure(figsize=(8,8), num="Ratio measured vs correct")
plt.plot(s_unique, ratio_perpendicular, label=r'$\frac{\xi_\perp^\mathrm{measured}}{\xi_\perp^\mathrm{correct}}$', linestyle='--', linewidth=0.6, marker='*', markersize=2, color='cyan')
plt.plot(s_unique, ratio_parallel, label=r'$\frac{\xi_\parallel^\mathrm{measured}}{\xi_\parallel^\mathrm{correct}}$', linestyle='--', linewidth=0.6, marker='*', markersize=2, color='magenta')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\frac{\xi^\mathrm{measured}}{\xi^\mathrm{correct}}$')
plt.title('Clustering Wedges Ratio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0,2)
plt.savefig("graphs/z1/clustWedge_ratio.pdf", dpi=600)

plt.show()