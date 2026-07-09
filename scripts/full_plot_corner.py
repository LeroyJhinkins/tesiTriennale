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
raw_what_stat = sys.argv[3]
what_stat_list = [stat for stat in raw_what_stat.split(',')]
raw_data_type = sys.argv[4]
data_type_list = [dt for dt in raw_data_type.split(',')]
raw_bins = sys.argv[5]
redshift_bins = [int(num) for num in raw_bins.split(',')]
isCluster = bool(int(sys.argv[6]))

if len(what_stat_list) != len(data_type_list):
    raise RuntimeError(f"{len(what_stat_list)} number of statistics given and {len(data_type_list)} number of data types given, must be the same")

z_mean_array = [1.0, 1.2, 1.4, 1.65]

if model == 'CLEFT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$1+b_1$", r"$b_2$", r"$b_{s^2}$", r"$\alpha_\xi$", r"$\alpha_v$", r"$\alpha_\sigma$"]
elif model == 'CLPT':
    params_list_full = [r"$h$", r"$A_s$", r"$\omega_c$", r"$1+b_1$", r"$b_2$", r"$\sigma_v$"]

truths = {r"$h$": 0.67, r"$A_s$": 2.11065, r"$\omega_c$": 0.271*0.67**2}

for z in redshift_bins:

    z_mean = z_mean_array[z - 1]

    samples = []
    colors = []
    labels = []
    styles = []
    args = []
    fills = []
    lws=[]

    for what_stat, data_type in zip(what_stat_list, data_type_list):

        if isCluster:
            name_1 = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive5000_pool_10_newcode_{data_type}.txt'
        else:
            name_1 = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive1000_pool_3_newcode_{data_type}.txt'

        loaded_samples = np.loadtxt(name_1)
        loaded_samples[:,5] += 1
        sample1 = MCSamples(samples=loaded_samples[:,2:], weights=loaded_samples[:,0], names=params_list_full)
        sample1.updateSettings({'contours': [0.68, 0.95]})
        samples.append(sample1)

        if what_stat == 'multipoles' and data_type == 'measured':
            color = '#E31A1C'
            line_style = '-'
            fill_contour = True
            label_text = f'{model} Multipoles (Measured): smin {smin}'
            
        elif what_stat == 'multipoles' and data_type != 'measured':
            color = "#9B0EBE"
            line_style = '--'
            fill_contour = False
            label_text = f'{model} Multipoles (Correct): smin {smin}'
            
        elif what_stat != 'multipoles' and data_type == 'measured':
            color = "#1660CF"
            line_style = '-'
            fill_contour = True
            label_text = f'{model} Wedges (Measured): smin {smin}'
            
        else:
            color = '#33A02C'
            line_style = '--'
            fill_contour = False
            label_text = f'{model} Wedges (Correct): smin {smin}'

        colors.append(color)
        fills.append(fill_contour)
        labels.append(label_text)
        styles.append(line_style)
        args.append({'lw': 1 if fill_contour else 1.3, 'color': color, 'ls': line_style})
        lws.append(2)

    g = plots.getSubplotPlotter(rc_sizes=True)
    g.settings.num_plot_contours = 2
    g.settings.figure_legend_frame = False
    g.settings.line_labels = True
    g.settings.axes_labelsize = 48
    if model == 'CLEFT':
        g.settings.axes_fontsize = 30
        g.settings.legend_fontsize = 33
    elif model == 'CLPT':
        g.settings.axes_fontsize = 23
        g.settings.legend_fontsize = 23
    g.settings.alpha_filled_add = 0.5

    g.triangle_plot(
        samples,
        params_list_full,
        markers= truths,
        marker_args={'lw': 1.2, 'color': '#404040', 'linestyle': ':'},
        contour_colors=colors,
        contour_ls=styles,
        contour_lws=lws,
        filled=fills,
        legend_labels=labels,
        line_args=args
    )

    stats = "".join([word[0].capitalize() for word in raw_what_stat.split(",")])
    types = "".join([word[0].capitalize() for word in raw_data_type.split(",")])

    if isCluster:
        out = f"graphs/z{z}_rebin/cluster/ELM_contours_{model}_smin_{smin}_{stats}_{types}.pdf"
    else:
        out = f"graphs/z{z}_rebin/ELM_contours_{model}_smin_{smin}_{stats}_{types}.pdf"

    g.export(out)



