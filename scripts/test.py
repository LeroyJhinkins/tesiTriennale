# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import math
# import emcee
# import corner
# from itertools import chain
# from astropy.table import Table
from astropy.io import fits
# plt.rcParams.update({'font.size': 14})
# plt.rcParams['text.usetex'] = True  # Abilita LaTeX vero
# plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # Carica siunitx

# try using pyqt for matplotlib.pyplot.show()
# try:
#     matplotlib.use("QtAgg")
# except ImportError:
#     print("QtAgg backend not available, using default backend.")
# matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


# Percorso del file FITS
filepath = "data/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_AUTO_REC_2DPOL_20250402T154727.0Z.fits"

print()
# Apri il file FITS usando un context manager
with fits.open(filepath) as hdul:
    
    # Mostra informazioni generali sulle HDU
    hdul.info()
    
    # Ottieni la tabella
    table_hdu = hdul[1]  # HDU 0 è un contenitore vuoto che precede la vera tabella
    table_data = table_hdu.data # type: ignore
    
    # Lista delle colonne
    print(table_data.columns.names)
    
    # Mostra le prime righe
    print(table_data[:5])

#     s_list = table_data['SCALE_1D']
#     mu_list = table_data['SCALE_2D']
#     xi_list = table_data['XI']

# print(type(mu_list))
# print(min(s_list), " ", max(s_list))

print()

# Percorso del file FITS
filepath = "data/m_z1_1_correct/data/EUC_LE3_GCL_2PCF__Correlation_MULTIPOLES_20250402T154727.0Z.fits"

# Apri il file FITS usando un context manager
with fits.open(filepath) as hdul:
    
    # Mostra informazioni generali sulle HDU
    hdul.info()
    
    # Ottieni la tabella
    table_hdu = hdul[1]  # HDU 0 è un contenitore vuoto che precede la vera tabella
    table_data = table_hdu.data # type: ignore
    
    # Lista delle colonne
    print(table_data.columns.names)
    
    # Mostra le prime righe
    print(table_data[:5])