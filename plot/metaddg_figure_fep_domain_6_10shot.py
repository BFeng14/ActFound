#!/usr/bin/env python
# coding: utf-8

import math
import warnings
import plot_utils
import plot_settings
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import json

sys.path.append(os.path.join(sys.path[0], '../'))
warnings.filterwarnings('ignore')

datasets = ["6-shot", "10-shot"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'transfer_qsar']
models_cvt = plot_settings.models_cvt

metric_name = "r2"
domain_name = "fep"
dataset_name = "chembl"

fepset = [{}, {}, {}, {}]
for x in models:
    with open(os.path.join(f"../test_results/result_fep/{domain_name}/{dataset_name}", x, "sup_num_6.json"), "r") as f:
        res = json.load(f)
    fepset[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fepset[0][x].append(math.sqrt(d))
        mean += d

    fepset[1][x] = []
    with open(os.path.join(f"../test_results/result_fep/{domain_name}/{dataset_name}", x, "sup_num_10.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if k == "thrombin":
            continue
        fepset[1][x].append(math.sqrt(d))
        mean += d



ax = plot_settings.get_wider_axis(double=True)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(2):
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(fepset[i][k]).tolist())
        stderrs.append(np.std(fepset[i][k] / np.sqrt(len(fepset[i][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)

min_val = np.min(np.array(means_all) - np.array(stderrs_all))
max_val = np.max(np.array(means_all) + np.array(stderrs_all))
min_val = 0. #max(min_val-(max_val-min_val)*0.15, 0.)

plot_legend = True
ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE(pK)"
ax = plot_settings.get_wider_axis(double=False)


means_all[0].insert(3, 0.6605)
means_all[1].insert(3, 0.6840)
labels.insert(3, "PBCNet")
colors.insert(3, '#35978f')
colors[4] = '#01665e'
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel='', ylabel="$\\rho_P$", color_legend=labels if plot_legend else None,
    nested_color=colors, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=4)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()
if metric_name == "r2":
    plt.axhline(y=0.7459, color='#b2182b', linestyle='--', lw=1)
elif metric_name == "rmse":
    plt.axhline(y=0.591, color='#b2182b', linestyle='--', lw=1)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ax.yaxis.set_major_locator(MultipleLocator(0.05))  
ax.set_ylim(0.45, 0.80)


ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  

plt.savefig(f'./figs/supplement.10.compare_with_PBCNet_rhp_P_{domain_name}_{dataset_name}.pdf')
plt.show()

print("finish")