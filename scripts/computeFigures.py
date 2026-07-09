import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_metrics(file_path, true_values):
    
    data = np.loadtxt(file_path)
    
    weights = data[:, 0]
    parameters = data[:, [2, 3, 4]] # shape: (N, 3)
    
    means = np.average(parameters, axis=0, weights=weights)
    
    cov = np.cov(parameters, aweights=weights, bias=False, ddof=1, rowvar=False)
    
    # Figure of Merit (FoM)
    det_cov = np.linalg.det(cov)
    fom = 1.0 / np.sqrt(det_cov)
    
    # Figure of Bias (FoB)
    delta = means - np.array(true_values)
    inv_cov = np.linalg.inv(cov)
    fob = np.sqrt(delta.T @ inv_cov @ delta)
    
    return fom, fob

def plot_FoM_FoB(model,s_min_list,z_mean_array):
    
    truths = [0.67, 2.11065, 0.271 * 0.67**2] # [h, A_s, omega_c]
    
    stats = ['multipoles', 'wedges']
    data_types = ['measured', 'correct']

    plot_data = {
        z_mean: {'FoM': {}, 'FoB': {}} for z_mean in z_mean_array
    }

    for z_mean in z_mean_array:
        
        for stat in stats:
            for dt in data_types:
                key = f"{stat}_{dt}"
                fom_array = []
                fob_array = []
                
                for smin in s_min_list:
                    if isCluster:
                        filename = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{stat}_nlive5000_pool_10_newcode_{dt}.txt'
                    else:
                        filename = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin{smin}-de_model_lambda-cov_it_2-what_stat_{stat}_nlive1000_pool_3_newcode_{dt}.txt'
                        
                    fom, fob = compute_metrics(filename, truths)
                        
                    fom_array.append(fom)
                    fob_array.append(fob)
                
                plot_data[z_mean]['FoM'][key] = fom_array
                plot_data[z_mean]['FoB'][key] = fob_array

    styles = {
        'multipoles_measured': {'color': '#E31A1C', 'linestyle': '-',  'marker': 'o', 'label': 'Multipoles Measured'},
        'multipoles_correct':  {'color': '#9B0EBE', 'linestyle': '--', 'marker': 'o', 'markerfacecolor': 'none', 'label': 'Multipoles Correct'},
        'wedges_measured':     {'color': '#1660CF', 'linestyle': '-',  'marker': 's', 'label': 'Wedges Measured'},
        'wedges_correct':      {'color': '#33A02C', 'linestyle': '--', 'marker': 's', 'markerfacecolor': 'none', 'label': 'Wedges Correct'}
    }

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey='row')
    
    fig.suptitle(f'Model: {model}', fontsize=27, fontweight='bold', y=0.98)

    fom_label = r'$\text{FoM}(h, A_s, \omega_c)$'
    fob_label = r'$\text{FoB}(h, A_s, \omega_c)$'

    for col, z_val in enumerate(z_mean_array):
        axs[0, col].set_title(f'$z = {z_val}$', fontsize=13, pad=10)

    for row, metric in enumerate(['FoM', 'FoB']):
        axs[row, 0].set_ylabel(fom_label if metric == 'FoM' else fob_label, fontsize=22)
        
        for col, z_val in enumerate(z_mean_array):
            ax = axs[row, col]
            
            for config_name, style_kwargs in styles.items():
                y_data = plot_data[z_val][metric][config_name]
                ax.plot(s_min_list, y_data, linewidth=2, markersize=7, **style_kwargs)
            

            if metric == 'FoM':
                ax.set_yscale('log')
                # ax.set_ylim(1e6, 1e9) # Adjust based on your typical FoM range
                ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
                
            else:
                # ax.set_ylim(0, 6) 
                ax.fill_between(s_min_list, 0, 1.87, color='grey', alpha=0.2) # corresponding to 1 sigma
                ax.fill_between(s_min_list, 1.87, 2.8, color='grey', alpha=0.08) # corresponding to 2 sigma
                    
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=18)
            ax.set_xticks(s_min_list)
            ax.set_xticklabels(s_min_list)

    for col in range(2):
        axs[1, col].set_xlabel(r'$s_{\text{min}} \, [h^{-1}\text{Mpc}]$', fontsize=22)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=True, fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    
    if isCluster:
        out = f"graphs/z{redshift_bins[0]}_rebin/cluster/ELM_contours_{model}_FoM_FoB.pdf"
    else:
        out = f"graphs/z{redshift_bins[0]}_rebin/ELM_contours_{model}_FoM_FoB.pdf"

    plt.savefig(out, bbox_inches='tight')

if __name__ == "__main__":
    
    model = sys.argv[1]
    raw_smin = sys.argv[2]
    smin_list = [int(smin) for smin in raw_smin.split(',')]
    raw_bins = sys.argv[3]
    redshift_bins = [int(num) for num in raw_bins.split(',')]
    isCluster = bool(int(sys.argv[4]))

    z_mean_list = [1.0, 1.2, 1.4, 1.65]
    z_mean_array = [z_mean_list[z-1] for z in redshift_bins]
    truths = [0.67, 2.11065, 0.271*0.67**2] # [h, A_s, omega_c]

    plot_FoM_FoB(model,smin_list,z_mean_array)

    # for z in redshift_bins:

    #     z_mean = z_mean_list[z - 1]

    #     for what_stat, data_type in zip(['multipoles','multipoles','wedges','wedges'], ['measured','correct','measured','correct']):

    #         if isCluster:
    #             filename = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin20-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive5000_pool_10_newcode_{data_type}.txt'
    #         else:
    #             filename = f'chains/{model}_lambda/chain_{model}-z_{z_mean}-smin20-de_model_lambda-cov_it_2-what_stat_{what_stat}_nlive1000_pool_3_newcode_{data_type}.txt'
            
    #         fom, fob = compute_metrics(filename, truths)
            
    #         print(f"\nz{z} {model} {what_stat.capitalize()} ({data_type}), smin = 20:")
    #         print(f"FoM: {fom:.4e}")
    #         print(f"FoB: {fob:.4f}")