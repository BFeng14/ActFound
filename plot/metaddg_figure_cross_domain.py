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


datasets = ["ChE to BDB", "BDB to ChE"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar']
models_cvt = plot_settings.models_cvt
metric_name = sys.argv[1]
if metric_name == "rmse":
    models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT']

import os
import json
bdb = {}
chembl = {}
for x in models:
    with open(os.path.join("../test_results/result_cross/bdb2chembl", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    bdb[x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        bdb[x].append(d)
        mean += d
        #bdb[x].append(mean / 10)
    chembl[x] = []
    with open(os.path.join("../test_results/result_cross/chembl2bdb", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        chembl[x].append(d)
        mean += d


ax = plot_settings.get_square_axis()
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
mean = {}
mean["chembl"], mean["bdb"] = {}, {}
std = {}
std["chembl"], std["bdb"] = {}, {}
for k in chembl:
    mean["chembl"][k] = np.mean(chembl[k])
    std["chembl"][k] = np.std(chembl[k] / np.sqrt(len(chembl[k])))
for k in bdb:
    mean["bdb"][k] = np.mean(bdb[k])
    std["bdb"][k] = np.std(bdb[k] / np.sqrt(len(bdb[k])))

means = []
stderrs = []
means.append([mean["chembl"][mod] for mod in models])
means.append([mean["bdb"][mod] for mod in models])
stderrs.append([std["chembl"][mod] for mod in models])
stderrs.append([std["bdb"][mod] for mod in models])

min_val = np.min(np.array(means) - np.array(stderrs))
max_val = np.max(np.array(means) + np.array(stderrs))
min_val = 0. #max(min_val-(max_val-min_val)*0.15, 0.)

plot_legend = False
ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"
ax = plot_settings.get_square_axis()
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
    ax.set_ylim(0.10, 0.35)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.50, 0.70)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/3.a.figure_cross_domain_{ylabel}.pdf')
elif ylabel == "RMSE":
    plt.savefig(f'./figs/3.b.figure_cross_domain_{ylabel}.pdf')

print("finish")

