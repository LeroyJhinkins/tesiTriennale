import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
from libs import ClusteringWedges as wdg
from libs import Multipoles as mp
from libs import BiMaps as bm
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
s_matrix_measured, mu_matrix_measured, dd_matrix_measured, dr_matrix_measured, rr_matrix_measured = read.readFITS_pairs_series_SMU("data/z1_data/z1_measured",nFiles, nElements)

print(f"Reading {nFiles} correct files...")
s_matrix_correct, mu_matrix_correct, dd_matrix_correct, dr_matrix_correct, rr_matrix_correct = read.readFITS_pairs_series_SMU("data/z1_data/z1_correct", nFiles, nElements)

print()
