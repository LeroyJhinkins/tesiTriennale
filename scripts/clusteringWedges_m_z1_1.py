import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libs import ReadFITS as read
from libs import Projections as prj
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

nWedges = 4
print(f"Calculating {nWedges} clustering wedges...")

s_unique = np.unique(s_array_correct)

wedges_measured = prj.compute_clusteringWedges(nWedges, s_array_measured, mu_array_measured, xi_array_measured)
wedges_correct = prj.compute_clusteringWedges(nWedges, s_array_correct, mu_array_correct, xi_array_correct)

prj.plot_clusteringWedges_measVScorr(nWedges, s_unique, wedges_measured, wedges_correct, base_path="graphs/m_z1_1")

ratio = prj.plot_clusteringWedges_ratio(nWedges, s_unique, wedges_measured, wedges_correct, base_path="graphs/m_z1_1")

plt.show()