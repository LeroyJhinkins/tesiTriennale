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


# read FITS file ------------------------------------------------------------------------------------------------------------------
# filepath_auto = "data/m_z1_1/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250403T181849.0Z.fits" # measured
filepath_auto = "data/m_z1_1/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits" # correct

s_array, mu_array, xi_array, nData = read.readFITS_auto(filepath_auto)
print("\nTotal number of points:", nData)


# Legendre projection -------------------------------------------------------------------------------------------------------------
# source: https://arxiv.org/pdf/1205.5573
# XI is a function of s and mu (:= cosθ)
# so in this dataset we have several values of XI(s, mu) -> for every value of s we have 200 values of XI, for -1 < mu < 1
# therefore we extract mu and XI values for every s
s_unique = np.unique(s_array)

l_values = np.array([0,1,2,3,4])
xi_multipoles = prj.compute_multipoles(l_values, s_array, mu_array, xi_array)

nPoints = 5
print(f"\nFirst {nPoints} points:")
prj.print_multipoles(l_values, s_array, xi_multipoles, nPoints)


# comparison with official data ---------------------------------------------------------------------------------------------------
# filepath_multipoles = "data/m_z1_1/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250403T181849.0Z.fits" # measured
filepath_multipoles = "data/m_z1_1/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250402T154727.0Z.fits" # correct

s_official, xi_official, _ = read.readFITS_multipoles(filepath_multipoles)
print("\nTotal number of official points:", len(s_official))
    
print(f"\nFirst {nPoints} official points (expanded in multipoles):")
prj.print_multipoles(l_values, s_official, xi_official, nPoints)

print("\nPrinting big differences (>10^-15) between calculations and official data: ")
diff = np.abs(xi_multipoles - xi_official)
for i in range(len(s_unique)):
    
    for j in range(len(l_values)):
    
        if diff[i, j] > 1.0e-15:
            print(f" Position ({i}, {j}): calculation = {xi_multipoles[i,j]}, official = {xi_official[i,j]}, difference = {diff[i,j]}")


# plot 1st monopole ---------------------------------------------------------------------------------------------------------------
prj.plot_multipole(0, s_unique, xi_multipoles[:,0], "graphs/m_z1_1")


# 2D map --------------------------------------------------------------------------------------------------------------------------
# plot in s and mu
# data in s and mu are already listed on a regular grid
# so we have nothing more to do except plotting the data
MU, S, XI = bm.plot_contourf("SMU", mu_array, s_array, xi_array, "graphs/m_z1_1", "measured", lvls=20)

# plot in r_p and pi
RP = S * np.sqrt(1 - MU**2)
PI = S * MU

plt.figure(figsize=(9,8), num="2Dmap (r_p, pi)")
contourRpPI = plt.contourf(RP, PI, np.log(np.abs(XI)), levels=30, cmap='turbo') # we use log to better observe differences in levels
                                                                                # abs is for avoiding log of negative numbers

cbarRpPI = plt.colorbar(contourRpPI, label=r'$\xi(r_p, \pi)$')

plt.xlim(0,30) # we want to zoom in to see redshift distortion
plt.ylim(-15,15) # which, as we can see, form almost circular levels of xi

plt.xlabel(r'$r_p \,[h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$\pi \,[h^{-1} \, \mathrm{Mpc}]$')
plt.title(r'2D map of $\xi(r_p, \pi)$')
plt.tight_layout()
plt.savefig("graphs/m_z1_1/contourfRpPI_meaasured.pdf", dpi=600)

plt.show()