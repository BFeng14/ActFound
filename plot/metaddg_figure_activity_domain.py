#!/usr/bin/env python
# coding: utf-8



import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import json

sys.path.append(os.path.join(sys.path[0], '../'))
import plot_settings
import plot_utils
import warnings
warnings.filterwarnings('ignore')


datasets = ["Activity type"]
models_cvt = plot_settings.models_cvt

metric_name = sys.argv[1]
shot_name = sys.argv[2]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']
if metric_name == "rmse":
    models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'RF', 'GPST', 'KNN']


import os
import json
bdb = {}
chembl = {}
for x in models:
    chembl[x] = []
    with open(os.path.join(f"../test_results/result_ood", x, f"sup_num_{shot_name}.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        chembl[x].append(d)
        mean += d

for x in chembl:
    print(len(chembl[x]))

ax = plot_settings.get_wider_axis(double=False)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
mean = {}
mean["chembl"], mean["bdb"] = {}, {}
std = {}
std["chembl"], std["bdb"] = {}, {}
for k in chembl:
    mean["chembl"][k] = np.mean(chembl[k])
    std["chembl"][k] = np.std(chembl[k] / np.sqrt(len(chembl[k])))

means = []
stderrs = []
means.append([mean["chembl"][mod] for mod in models])
stderrs.append([std["chembl"][mod] for mod in models])

min_val = np.min(np.array(means) - np.array(stderrs))
max_val = np.max(np.array(means) + np.array(stderrs))
min_val = 0.0#max(min_val-(max_val-min_val)*0.15, 0.)

ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"

# ax = plot_settings.get_wider_axis(False)
plt.figure(figsize=(int(plot_settings.FIG_WIDTH * 0.75), plot_settings.FIG_HEIGHT))
ax = plt.subplot(1, 1, 1)

plot_legend = False
plot_utils.grouped_barplot(
        ax, means, 
        datasets,
        xlabel='', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=0, anchorpoint='center',
        legend_loc='upper left',
        min_val=min_val, scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if ylabel == "r2":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.05, 0.35)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.03))  
    ax.set_ylim(0.57, 0.72)
# ax.set_title('Cross-unit prediction', size=15)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/2.g.figure_activity_domain_{ylabel}.pdf')
elif ylabel == "RMSE":
    plt.savefig(f'./figs/2.h.figure_activity_domain_{ylabel}.pdf')


print("finish")
