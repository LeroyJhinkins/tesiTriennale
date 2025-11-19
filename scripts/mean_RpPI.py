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
nElements = 40000

print("\nReading measured files...", end=" ")
rp_matrix_measured, pi_matrix_measured, xi_matrix_measured = read.readFITS_auto_series_RpPI("data/measurements_pre_rec/m_z1_measured", nElements, kind="measured")

print("\nReading correct files...", end=" ")
rp_matrix_correct, pi_matrix_correct, xi_matrix_correct = read.readFITS_auto_series_RpPI("data/measurements_pre_rec/m_z1_correct", nElements, kind="correct")


# compute mean --------------------------------------------------------------------------------------------------------------------
# we want to get rid of statistical errors in order to emphasize the systematical error
# so we want to mean the values over the files
# computing the mean per column with axis=0 => this gives an array nElements long
rpMean_array_measured = np.mean(rp_matrix_measured, axis=0) # this is probably useless
piMean_array_measured = np.mean(pi_matrix_measured, axis=0) # just a check
xiMean_array_measured = np.mean(xi_matrix_measured, axis=0)
# xiStd_array_measured = np.std(xi_matrix_measured, axis=0)

rpMean_array_correct = np.mean(rp_matrix_correct, axis=0) # this is probably useless
piMean_array_correct = np.mean(pi_matrix_correct, axis=0) # just a check
xiMean_array_correct = np.mean(xi_matrix_correct, axis=0)
# xiStd_array_correct = np.std(xi_matrix_correct, axis=0)


# contour plot --------------------------------------------------------------------------------------------------------------------
if np.allclose(rpMean_array_measured, rpMean_array_correct, atol=1e-4) and \
    np.allclose(piMean_array_measured, piMean_array_correct, atol=1e-4): # this should always be true

    rp_unique = np.unique(rpMean_array_measured)
    pi_unique = np.unique(piMean_array_measured)
    
    RP, PI = np.meshgrid(rp_unique, pi_unique)
    XI_MEASURED = xiMean_array_measured.reshape((len(pi_unique), len(rp_unique))).T
    XI_CORRECT = xiMean_array_correct.reshape((len(pi_unique), len(rp_unique))).T

    # plot measured
    plt.figure(figsize=(10,8), num="2D map (r_p, pi) measured")
    contourRpPI_measured = plt.contourf(RP, PI, XI_MEASURED, levels=20, cmap='turbo')
    
    cbarRpPI_measured = plt.colorbar(contourRpPI_measured, label=r'$\xi^\mathrm{measured}(r_p, \pi)$')
    xi_measured_ticks = np.linspace(np.min(xiMean_array_measured), np.max(xiMean_array_measured), 9)
    cbarRpPI_measured.set_ticks(xi_measured_ticks)
    cbarRpPI_measured.set_ticklabels([f"{tick:.2f}" for tick in xi_measured_ticks])

    rp_ticks = np.linspace(np.min(rp_unique), np.max(rp_unique), 6)
    pi_ticks = np.linspace(np.min(pi_unique), np.max(pi_unique), 6)
    plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
    plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

    plt.xlim(0, 40)
    plt.ylim(0, 40)

    plt.xlabel(r'$r_\mathrm{p} \,(h^{-1} \, \mathrm{Mpc})$')
    plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
    plt.title(r'2D map of $\xi^\mathrm{measured}(r_p, \pi)$')
    plt.tight_layout()
    plt.savefig("graphs/measurements_pre_rec/2DmapRpPI_measured.pdf", dpi=600)

    # plot correct
    plt.figure(figsize=(10,8), num="2D map (r_p, pi) correct")
    contourRpPI_correct = plt.contourf(RP, PI, XI_CORRECT, levels=20, cmap='turbo')
    
    cbarRpPI_corret = plt.colorbar(contourRpPI_correct, label=r'$\xi^\mathrm{correct}(r_p, \pi)$')
    xi_ticks_correct = np.linspace(np.min(xiMean_array_correct), np.max(xiMean_array_correct), 9)
    cbarRpPI_corret.set_ticks(xi_ticks_correct)
    cbarRpPI_corret.set_ticklabels([f"{tick:.2f}" for tick in xi_ticks_correct])

    plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
    plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])
    
    plt.xlim(0, 40)
    plt.ylim(0, 40)

    plt.xlabel(r'$r_\mathrm{p} \,(h^{-1} \, \mathrm{Mpc})$')
    plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
    plt.title(r'2D map of $\xi^\mathrm{correct}(r_p, \pi)$')
    plt.tight_layout()
    plt.savefig("graphs/measurements_pre_rec/2DmapRpPI_correct.pdf", dpi=600)

else:
    raise RuntimeError("r_p and pi from measured and correct files are not compatible")

# plot ratio ----------------------------------------------------------------------------------------------------------------------
# we want to see the effects of interlopers
# so we compute measured/correct ratio to tell the difference between the two
# plot in r_p and pi

xi_ratio = xiMean_array_measured / xiMean_array_correct
XI_RATIO = xi_ratio.reshape(len(rp_unique), len(pi_unique))

mask = np.abs(XI_RATIO) > 2.1
XI_RATIO[mask] = np.nan

plt.figure(figsize=(10,8), num="Ratio (r_p, pi)")
contourRpPI_ratio = plt.imshow(XI_RATIO, cmap='turbo', vmin=-1.1, vmax=1.5, origin="lower", interpolation="bicubic")

# cbarRpPI_ratio = plt.colorbar(contourRpPI_ratio, label=r'$\log \left( \frac{\xi^\mathrm{measured}}{\xi^\mathrm{correct}} \right)$')
# xi_ratio_ticks = np.linspace(np.min(xi_ratio), np.max(xi_ratio), 9)
# cbarRpPI_ratio.set_ticks(xi_ratio_ticks)
# cbarRpPI_ratio.set_ticklabels([f"{tick:.2f}" for tick in xi_ratio_ticks])
plt.colorbar()

plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

# plt.xlim(0, 15)
# plt.ylim(0, 15)

plt.xlabel(r'$r_\mathrm{p} \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\log \left( \frac{\xi^\mathrm{measured}}{\xi^\mathrm{correct}}(r_\mathrm{p},\pi) \right)$')
plt.tight_layout()
plt.savefig("graphs/measurements_pre_rec/2DmapRpPI_ratio.pdf", dpi=600)



plt.show()