import math
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['text.usetex'] = True  # use real LaTeX
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

# try using pyqt for matplotlib.pyplot.show()
# try:
#     matplotlib.use("QtAgg")
# except ImportError:
#     print("QtAgg backend not available, using default backend.")
# matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


# read FITS file ------------------------------------------------------------------------------------------------------------------
filepath_auto = "data/m_z1_1_measured/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250403T181849.0Z.fits" # measured
#filepath_auto = "data/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits" # correct

with fits.open(filepath_auto) as hdul:
    
    # print()
    # hdul.info()
    
    table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
    table_data = table_hdu.data # type: ignore
                                # comment to ignore Pylance warning

    names = table_data.columns.names
    s_array = table_data[names[0]]
    mu_array = table_data[names[1]]
    xi_array = table_data[names[2]]
    
    print("\nTotal number of points:", len(s_array))


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