import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.io import fits
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = True  # use real LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

try:
    matplotlib.use("QtAgg")
except ImportError:
    print("QtAgg backend not available, using default backend.")
matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


# read FITS file ------------------------------------------------------------------------------------------------------------------
filepath_auto = "data/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250403T181849.0Z.fits" # measured
#filepath_auto = "data/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits" # correct

with fits.open(filepath_auto) as hdul:
    
    # print()
    # hdul.info()
    
    table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
    table_data = table_hdu.data # type: ignore
                                # comment to ignore Pylance warning
    nData = table_data.shape[0]
    nColumns = len(table_data.columns)

    names = table_data.columns.names
    s_array = table_data[names[0]]
    mu_array = table_data[names[1]]
    xi_array = table_data[names[2]]
    
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
filepath_multipoles = "data/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250403T181849.0Z.fits" # measured
#filepath_multipoles = "data/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250402T154727.0Z.fits" # correct

with fits.open(filepath_multipoles) as hd:
    
    # print()
    # hdul.info()
    
    table_hdu = hd[1]  # HDU 0 is an empty header that precedes the actual table
    table_data = table_hdu.data # type: ignore
                                # comment to ignore Pylance warning

    names = table_data.columns.names
    s_official = table_data[names[0]]
    xi_official = np.column_stack((table_data[names[1]], table_data[names[2]], table_data[names[3]], table_data[names[4]],
                                   table_data[names[5]]))

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

plt.figure(figsize=(13,8), num="2Dmap (mu, s)")
contourMUS = plt.contourf(MU, S, XI, levels=20, cmap='turbo')
# plt.xlim(0, np.max(mu_unique))

cbarMUS = plt.colorbar(contourMUS, label=r'$\xi(s,\mu)$')
xi_ticks = np.linspace(np.min(xi_array), np.max(xi_array), 9)
cbarMUS.set_ticks(xi_ticks)
cbarMUS.set_ticklabels([f"{tick:.3f}" for tick in xi_ticks])

s_ticks = np.linspace(np.min(s_unique), np.max(s_unique), 5)
mu_ticks = np.linspace(np.min(mu_unique), np.max(mu_unique), 6)
plt.xticks(mu_ticks, [f"{tick:.1f}" for tick in mu_ticks])
plt.yticks(s_ticks, [f"{tick:.0f}" for tick in s_ticks])

plt.xlabel(r'$\mu$')
plt.ylabel(r'$s \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\xi(\mu,s)$')

# plot in r_p and pi
# when we transform (mu, s) into (r_p, pi), the grid's regularity falls apart
# therefore we want to make a new regular grid in (r_p, pi)
# and interpolate xi on the new regular grid (using scipy)
# r_p_array = s_array * np.sqrt(1 - mu_array**2)
# pi_array  = s_array * mu_array


# rp_reg = np.linspace(np.min(r_p_array), np.max(r_p_array), 200) # new regular grid
# pi_reg = np.linspace(np.min(pi_array), np.max(pi_array), 200) # in r_p and pi
# RP_grid, PI_grid = np.meshgrid(rp_reg, pi_reg)

# XI_interp = griddata(
#     (r_p_array, pi_array),   # original irregular coordinates
#     xi_array,                # interpulating xi values
#     (RP_grid, PI_grid),      # new regular grid
#     method='linear'
# )

RP = S * np.sqrt(1 - MU**2)
PI = S * MU

plt.figure(figsize=(13,8), num="2Dmap (r_p, pi)")
# contourRpPI = plt.contourf(RP_grid, PI_grid, XI_interp, levels=20, cmap='turbo')
contourRpPI = plt.contourf(RP, PI, XI, levels=20, cmap='turbo')

cbarRpPI = plt.colorbar(contourRpPI, label=r'$\xi(r_p, \pi)$')
# xi_interpol_ticks = np.linspace(np.min(XI_interp), np.max(XI_interp), 9)
xi_interpol_ticks = np.linspace(np.min(XI), np.max(XI), 9)
cbarRpPI.set_ticks(xi_interpol_ticks)
cbarRpPI.set_ticklabels([f"{tick:.3f}" for tick in xi_interpol_ticks])

# rp_ticks = np.linspace(np.min(r_p_array), np.max(r_p_array), 10)
# pi_ticks = np.linspace(np.min(pi_array), np.max(pi_array), 10)
rp_ticks = np.linspace(np.min(RP), np.max(RP), 10)
pi_ticks = np.linspace(np.min(PI), np.max(PI), 10)
plt.xticks(rp_ticks, [f"{tick:.1f}" for tick in rp_ticks])
plt.yticks(pi_ticks, [f"{tick:.0f}" for tick in pi_ticks])

plt.xlabel(r'$r_p \,(h^{-1} \, \mathrm{Mpc})$')
plt.ylabel(r'$\pi \,(h^{-1} \, \mathrm{Mpc})$')
plt.title(r'2D map of $\xi(r_p, \pi)$')
plt.tight_layout()

plt.show()

# Define regular polar grid
# nr, nt = 150, 200
# rho = np.linspace(0, 3, nr)          # radial coordinate
# theta = np.linspace(0, 2*np.pi, nt)  # angular coordinate
# S, MU = np.meshgrid(s_unique, np.unique(mu_array))

# # Define scalar field F(rho, theta)
# F = np.cos(3*T) * np.exp(-R**2)

# # Convert to Cartesian coordinates for plotting
# X = R * np.cos(T)
# Y = R * np.sin(T)
 
# # Plot filled contours
# plt.figure(figsize=(13,8), num="2Dmap")
# contour = plt.contourf(X, Y, F, levels=30, cmap='plasma')
# plt.colorbar(contour, label='F(rho, theta)')
 
# # Optionally overlay contour lines
# lines = plt.contour(X, Y, F, levels=10, colors='k', linewidths=0.5)
# plt.clabel(lines, inline=True, fontsize=8)
 
# # Labels and layout
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Iso-contours of F(ρ, θ)')
# plt.axis('equal')
# plt.show()



