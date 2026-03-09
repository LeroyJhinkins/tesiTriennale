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
model = sys.argv[1]
smin = sys.argv[2]
what_stat = sys.argv[3]
data_type = sys.argv[4]
raw_bins = sys.argv[5]
redshift_bins = [int(num) for num in raw_bins.split(',')]

z_mean_array = [1.0, 1.2, 1.4, 1.65]

if model == 'CLEFT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$b_1$", r"$b_2$", r"$bs2$", r"$a_x$", r"$a_v$", r"$a_s$"]
elif model == 'CLPT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$b_1$", r"$b_2$", r"$\sigma_v$"]

for z in redshift_bins:
    z_mean = z_mean_array[z - 1]
    name_1 = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive1000_pool_3_newcode_{data_type}.txt'

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

    g.triangle_plot([sample1], params_list_full, contour_colors=['tab:red', 'tab:blue', 'tab:green'], filled=True, legend_labels=[f'{model} {what_stat} {data_type}: smin {smin}'], line_args=[{'lw':2, 'color':'tab:purple'},{'lw':2, 'color':'tab:blue'},{'lw':2, 'color':'tab:green'}])

    g.subplots[0,0].axvline(0.67, color='grey', linestyle='--', lw=2) # type: ignore
    g.subplots[1,0].axhline(2.11065, color='grey', linestyle='--', lw=2) # type: ignore
    g.subplots[1,1].axvline(2.11065, color='grey', linestyle='--', lw=2) # type: ignore
    g.subplots[2,2].axvline(0.271*0.67**2, color='grey', linestyle='--', lw=2) # type: ignore
    g.subplots[2,0].axhline(0.271*0.67**2, color='grey', linestyle='--', lw=2) # type: ignore
    g.subplots[2,1].axhline(0.271*0.67**2, color='grey', linestyle='--', lw=2) # type: ignore

    for counter_ax in range(1,len(params_list_full)):
        g.subplots[counter_ax,0].axvline(0.67, color='grey', linestyle='--', lw=2) # type: ignore
    for counter_ax in range(2,len(params_list_full)):
        g.subplots[counter_ax,1].axvline(2.11065, color='grey', linestyle='--', lw=2) # type: ignore
    for counter_ax in range(3,len(params_list_full)):
        g.subplots[counter_ax,2].axvline(0.271*0.67**2, color='grey', linestyle='--', lw=2) # type: ignore


    out = f"graphs/z{z}_rebin/ELM_contours_{model}_smin_{smin}_{what_stat}_{data_type}.pdf"
    g.export(out)
