#!/usr/bin/python

from getdist import plots, MCSamples
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import sys

rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

working_dir = "chains/CLEFT_lambda/"
smin = sys.argv[1]
params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$b_1$", r"$b_2$", r"$bs2$", r"$a_x$", r"$a_v$", r"$a_s$"]

for model in ['CLEFT']:
    for z in range(1,2):
        name_1 = f'{working_dir}chain_{model}-z_{z}.0-smin{smin}-de_model_lambda-cov_it_2-what_stat_multipoles_nlive1000_pool_4_newcode.txt'

        loaded_samples = np.loadtxt(name_1)
        sample1 = MCSamples(samples=loaded_samples[:,2:], weights=loaded_samples[:,0], names=params_list_full)
        sample1.updateSettings({'contours': [0.68, 0.95]})

        g = plots.getSubplotPlotter(rc_sizes=True)
        g.settings.num_plot_contours = 3
        g.settings.figure_legend_frame = False
        g.settings.line_labels = True
        g.settings.axes_fontsize = 30
        g.settings.axes_labelsize = 45
        g.settings.legend_fontsize = 50

        g.triangle_plot([sample1], params_list_full, contour_colors=['tab:red', 'tab:blue', 'tab:green'], filled=True, legend_labels=[f'{model} multipoles'], line_args=[{'lw':2, 'color':'tab:purple'},{'lw':2, 'color':'tab:blue'},{'lw':2, 'color':'tab:green'}])

        g.subplots[0,0].axvline(0.67, color='grey', linestyle='--', lw=2)
        g.subplots[1,0].axhline(2.11065, color='grey', linestyle='--', lw=2)
        g.subplots[1,1].axvline(2.11065, color='grey', linestyle='--', lw=2)
        g.subplots[2,2].axvline(0.271*0.67**2, color='grey', linestyle='--', lw=2)
        g.subplots[2,0].axhline(0.271*0.67**2, color='grey', linestyle='--', lw=2)
        g.subplots[2,1].axhline(0.271*0.67**2, color='grey', linestyle='--', lw=2)

        for counter_ax in range(1,9):
            g.subplots[counter_ax,0].axvline(0.67, color='grey', linestyle='--', lw=2)
        for counter_ax in range(2,9):
            g.subplots[counter_ax,1].axvline(2.11065, color='grey', linestyle='--', lw=2)
        for counter_ax in range(3,9):
            g.subplots[counter_ax,2].axvline(0.271*0.67**2, color='grey', linestyle='--', lw=2)


        out = f"graphs/z{z}_rebin/ELM_contours_{model}_smin_{smin}_multipoles.pdf"
        g.export(out)
