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

# read measured and correct FITS files --------------------------------------------------------------------------------------------
filepath_auto_measured = "data/m_z1_1/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250403T181849.0Z.fits" # measured
filepath_auto_correct = "data/m_z1_1/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits" # correct

s_array_measured, mu_array_measured, xi_array_measured, nData_measured = read.readFITS_auto(filepath_auto_measured)
s_array_correct, mu_array_correct, xi_array_correct, nData_correct = read.readFITS_auto(filepath_auto_correct)
print("\nTotal number of points:", nData_measured)

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

xi_perpendicular_measured, xi_parallel_measured = clusteringWedges(s_array_measured, mu_array_measured, xi_array_measured)
xi_perpendicular_correct, xi_parallel_correct = clusteringWedges(s_array_correct, mu_array_correct, xi_array_correct)

ratio_perpendicular = xi_perpendicular_measured / xi_perpendicular_correct
ratio_parallel = xi_parallel_measured / xi_parallel_correct

s_unique = np.unique(s_array_correct)

plt.figure(figsize=(6,6), num="Perpendicular")
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_measured, label=r'$\xi_\perp^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='o', markersize=2, color='blue')
plt.plot(s_unique, (s_unique**2) * xi_perpendicular_correct, label=r'$\xi_\perp^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='s', markersize=2, color='green')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$s^2 \xi_\perp$')
plt.title('Perpendicular wedge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/m_z1_1/clustWedge_perp.pdf", dpi=600)


plt.figure(figsize=(6,6), num="Parallel")
plt.plot(s_unique, (s_unique**2) * xi_parallel_measured, label=r'$\xi_\parallel^\mathrm{measured}(s)$', linestyle='--', linewidth=0.6, marker='o', markersize=2, color='orange')
plt.plot(s_unique, (s_unique**2) * xi_parallel_correct, label=r'$\xi_\parallel^\mathrm{correct}(s)$', linestyle='--', linewidth=0.6, marker='s', markersize=2, color='red')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$s^2 \xi_\parallel$')
plt.title('Parallel wedge')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/m_z1_1/clustWedge_paral.pdf", dpi=600)


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
plt.savefig("graphs/m_z1_1/clustWedge_perpVSparal.pdf", dpi=600)


plt.figure(figsize=(8,8), num="Ratio measured vs correct")
plt.plot(s_unique, ratio_perpendicular, label=r'$\frac{\xi_\perp^\mathrm{measured}}{\xi_\perp^\mathrm{correct}}$', linestyle='--', linewidth=0.6, marker='*', markersize=2, color='cyan')
plt.plot(s_unique, ratio_parallel, label=r'$\frac{\xi_\parallel^\mathrm{measured}}{\xi_\parallel^\mathrm{correct}}$', linestyle='--', linewidth=0.6, marker='*', markersize=2, color='magenta')

plt.xlabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\frac{\xi^\mathrm{measured}}{\xi^\mathrm{correct}}$')
plt.title('Clustering Wedges Ratio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("graphs/m_z1_1/clustWedge_ratio.pdf", dpi=600)

plt.show()