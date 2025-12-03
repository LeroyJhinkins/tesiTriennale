import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
from libs import BiMaps as bm
from libs import ClusteringWedges as wdg
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
rp_matrix_measured, pi_matrix_measured, dd_matrix_measured, dr_matrix_measured, rr_matrix_measured = read.readFITS_pairs_series_RpPI("data/measurements_pre_rec/m_z1_measured",
                                                                                                                                     nElements,
                                                                                                                                     kind="measured")

print("Reading correct files...", end=" ")
rp_matrix_correct, pi_matrix_correct, dd_matrix_correct, dr_matrix_correct, rr_matrix_correct = read.readFITS_pairs_series_RpPI("data/measurements_pre_rec/m_z1_correct",
                                                                                                                                nElements,
                                                                                                                                kind="correct")
