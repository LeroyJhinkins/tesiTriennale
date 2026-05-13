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

model = sys.argv[1]
smin = sys.argv[2]
what_stat = sys.argv[3]
data_type = sys.argv[4]
raw_bins = sys.argv[5]
redshift_bins = [int(num) for num in raw_bins.split(',')]
if len(redshift_bins) != 2:
    raise RuntimeError(f"{len(redshift_bins)} redshift bins given... Must give 2")

isCluster = True

z_mean_array = [1.0, 1.2, 1.4, 1.65]

if model == 'CLEFT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$b_1$", r"$b_2$", r"$bs2$", r"$a_x$", r"$a_v$", r"$a_s$"]
elif model == 'CLPT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$b_1$", r"$b_2$", r"$\sigma_v$"]

params_cosmo = [r"$h$", r"$A_s$", r"$\omega_c$"]
truths = {r"$h$": 0.67, r"$A_s$": 2.11065, r"$\omega_c$": 0.271*0.67**2}
scale = {r"$h$": [0.63, 0.715], r"$A_s$": [1.2, 3.0], r"$\omega_c$": [0.09, 0.155]}

if isCluster:
    path_1 = f'chains/{model}_lambda/chain_{model}-z_{z_mean_array[redshift_bins[0]-1]}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive5000_pool_10_newcode_{data_type}.txt'
else:
    path_1 = f'chains/{model}_lambda/chain_{model}-z_{z_mean_array[redshift_bins[0]-1]}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive1000_pool_3_newcode_{data_type}.txt'
data_1 = np.loadtxt(path_1)
sample_1 = MCSamples(samples=data_1[:,2:], weights=data_1[:,0], names=params_list_full)
sample_1.updateSettings({'contours': [0.68, 0.95]})
if isCluster:
    path_2 = f'chains/{model}_lambda/chain_{model}-z_{z_mean_array[redshift_bins[1]-1]}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive5000_pool_10_newcode_{data_type}.txt'
else:
    path_2 = f'chains/{model}_lambda/chain_{model}-z_{z_mean_array[redshift_bins[1]-1]}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive1000_pool_3_newcode_{data_type}.txt'

data_2 = np.loadtxt(path_2)
sample_2 = MCSamples(samples=data_2[:,2:], weights=data_2[:,0], names=params_list_full)
sample_2.updateSettings({'contours': [0.68, 0.95]})

g = plots.getSubplotPlotter(rc_sizes=True)
g.settings.line_labels = True
g.settings.axes_fontsize = 14
g.settings.axes_labelsize = 21
g.settings.legend_fontsize = 13

g.triangle_plot(
    [sample_1, sample_2], 
    params=params_cosmo,
    param_limits=scale,
    markers=truths,
    marker_args = {'lw': 1.3, 'color': 'grey', 'linestyle': '--'},
    filled=True,
    contour_colors=['tab:red', 'tab:blue'],
    legend_labels=[f'{model} z{redshift_bins[0]} {what_stat} {data_type}: smin {smin}', f'{model} z{redshift_bins[1]} {what_stat} {data_type}: smin {smin}'],
    line_args=[{'lw':1.5, 'color':'tab:red'}, {'lw':1.5, 'color':'tab:blue'}],
)

if isCluster:
    out = f"graphs/z{redshift_bins[0]}_rebin/cluster/ELM_contours_{model}_z{redshift_bins[0]}_vs_z{redshift_bins[1]}_smin_{smin}_{what_stat}_{data_type}.pdf"
else:
    out = f"graphs/z{redshift_bins[0]}_rebin/ELM_contours_{model}_z{redshift_bins[0]}_vs_z{redshift_bins[1]}_smin_{smin}_{what_stat}_{data_type}.pdf"
g.export(out)