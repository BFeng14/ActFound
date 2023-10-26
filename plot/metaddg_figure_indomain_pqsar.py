#!/usr/bin/env python
# coding: utf-8

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

datasets = ["pQSAR-ChEMBL"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']
models_cvt = plot_settings.models_cvt


metric_name = sys.argv[1]
if metric_name == "rmse":
    models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'RF', 'GPST', 'KNN']
elif metric_name == "R2os":
    models = ['actfound_fusion', 'actfound_transfer', 'protonet', 'DKT', 'RF', 'GPST']

fsmol = [{}]
for x in models:
    if not os.path.exists(os.path.join("../test_results/result_indomain/pqsar", x)):
        continue
    with open(os.path.join("../test_results/result_indomain/pqsar", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d


# ax = plot_settings.get_wider_axis(double=True)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(1):
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(fsmol[i][k]))
        stderrs.append(np.std(fsmol[i][k] / np.sqrt(len(fsmol[i][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)

min_val = np.min(np.array(means_all) - np.array(stderrs_all))
max_val = np.max(np.array(means_all) + np.array(stderrs_all))
min_val = max(min_val-(max_val-min_val)*0.15, 0.)

ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"
plt.figure(figsize=(int(plot_settings.FIG_WIDTH * 0.75), plot_settings.FIG_HEIGHT))
ax = plt.subplot(1, 1, 1)
plot_legend = False
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel='', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
    nested_color=colors, nested_errs=stderrs_all, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if ylabel == "r2":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.15, 0.50)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.65, 1.05)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=4)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/2.d.figure_pqsar_indomain_{ylabel}.pdf')
elif ylabel == "RMSE":
    plt.savefig(f'./figs/2.e.figure_pqsar_indomain_{ylabel}.pdf')