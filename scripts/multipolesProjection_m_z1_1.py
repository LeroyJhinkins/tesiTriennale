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


# read FITS file ------------------------------------------------------------------------------------------------------------------
# filepath_auto = "data/m_z1_1/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250403T181849.0Z.fits" # measured
filepath_auto = "data/m_z1_1/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits" # correct

s_array, mu_array, xi_array, nData = read.readFITS_auto(filepath_auto)
print("\nTotal number of points:", nData)


# Legendre projection -------------------------------------------------------------------------------------------------------------
# source: https://arxiv.org/pdf/1205.5573
def legendre(l, mu):
    
    coeffs = np.zeros(l + 1)
    coeffs[l] = 1.0
    return np.polynomial.legendre.Legendre(coeffs)(mu)

# XI is a function of s and mu (:= cosθ)
# so in this dataset we have several values of XI(s, mu) -> for every value of s we have 200 values of XI, for -1 < mu < 1
# therefore we extract mu and XI values for every s
s_unique = np.unique(s_array)

l_values = [0,1,2,3,4]
xi_multipoles = np.zeros((len(s_unique), len(l_values)))

for i, s in enumerate(s_unique):
        
    mask = (s_array == s)
    mu_vals = mu_array[mask]
    xi_vals = xi_array[mask]

    delta_mu = np.float64(0.01)

    for j, l in enumerate(l_values):
            
        P_l_array = legendre(l, mu_vals)
        xi_multipoles[i,j] = ((2*l + 1)/2 * np.sum(xi_vals * P_l_array) * delta_mu) # l-th coefficient of the Legendre projection
                                                                                    # for the i-th value of XI

print("\nFirst five points:")
print("s XI0 XI1 XI2 XI3 XI4")
for s, xi in zip(s_unique[:5], xi_multipoles[:,:5]):
    xi_str = ", ".join(f"{x:.10e}" for x in xi)
    print(s, xi_str)


# comparison with official data ---------------------------------------------------------------------------------------------------
# filepath_multipoles = "data/m_z1_1/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250403T181849.0Z.fits" # measured
filepath_multipoles = "data/m_z1_1/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250402T154727.0Z.fits" # correct

s_official, xi_official, _ = read.readFITS_multipoles(filepath_multipoles)
print("\nTotal number of official points:", len(s_official))
    
print("\nFirst five official points (expanded in multipoles):")
print("s XI0 XI1 XI2 XI3 XI4")
for s, xi in zip(s_official[:5], xi_official[:,:5]):
    xi_str = ", ".join(f"{x:.10e}" for x in xi)
    print(s, xi_str)

print("\nPrinting big differences (>10^-15) between calculations and official data: ")
diff = np.abs(xi_multipoles - xi_official)
for i in range(len(s_unique)):
    
    for j in range(len(l_values)):
    
        if diff[i, j] > 1.0e-15:
            print(f" Position ({i}, {j}): calculation = {xi_multipoles[i,j]}, official = {xi_official[i,j]}, difference = {diff[i,j]}")


# 2D map --------------------------------------------------------------------------------------------------------------------------
# plot in s and mu
# data in s and mu are already listed on a regular grid
# so we have nothing more to do except plotting the data
mu_unique = np.unique(mu_array)
MU, S = np.meshgrid(mu_unique, s_unique)
XI = xi_array.reshape(len(s_unique), len(mu_unique))

plt.figure(figsize=(9,8), num="2Dmap (mu, s)")
contourMUS = plt.contourf(MU, S, XI, levels=20, cmap='turbo')

# plt.xlim(0, np.max(mu_unique))
# plt.ylim(0,10)

cbarMUS = plt.colorbar(contourMUS, label=r'$\xi(s,\mu)$')
xi_ticks = np.linspace(np.min(xi_array), np.max(xi_array), 9)
cbarMUS.set_ticks(xi_ticks)
cbarMUS.set_ticklabels([f"{tick:.3f}" for tick in xi_ticks])

s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 6)
mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 5)
plt.xticks(mu_ticks, [f"{tick:.1f}" for tick in mu_ticks])
plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])

plt.xlabel(r'$\mu$')
plt.ylabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\xi(\mu,s)$')
plt.savefig("graphs/m_z1_1/contourfSMU.pdf", dpi=600)

# plot in r_p and pi
RP = S * np.sqrt(1 - MU**2)
PI = S * MU

plt.figure(figsize=(9,8), num="2Dmap (r_p, pi)")
contourRpPI = plt.contourf(RP, PI, np.log(np.abs(XI)), levels=30, cmap='turbo') # we use log to better observe differences in levels
                                                                                # abs is for avoiding log of negative numbers

cbarRpPI = plt.colorbar(contourRpPI, label=r'$\xi(r_p, \pi)$')
xi_interpol_ticks = np.linspace(np.min(np.log(np.abs(XI))), np.max(np.log(np.abs(XI))), 9)
cbarRpPI.set_ticks(xi_interpol_ticks)
cbarRpPI.set_ticklabels([f"{tick:.3f}" for tick in xi_interpol_ticks])

rp_ticks = np.linspace(np.min(RP), np.max(RP), 30)
pi_ticks = np.linspace(np.min(PI), np.max(PI), 30)
plt.xticks(rp_ticks, [f"{tick:.0f}" for tick in rp_ticks])
plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

plt.xlim(0,30) # we want to zoom in to see redshift distortion
plt.ylim(-15,15) # which, as we can see, form almost circular levels of xi

plt.xlabel(r'$r_p \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\xi(r_p, \pi)$')
plt.tight_layout()
plt.savefig("graphs/m_z1_1/contourfRpPI.pdf", dpi=600)

plt.show()