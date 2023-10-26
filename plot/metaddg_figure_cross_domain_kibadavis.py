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


metric_name = sys.argv[1]
domain_name = sys.argv[2]
shot = 16

datasets = [f"ChE to {domain_name}", f"BDB to {domain_name}"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar']
models_cvt = plot_settings.models_cvt

if metric_name == "rmse":
    models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT']

import os
import json
import math
bdb = {}
chembl = {}
for x in models:
    with open(os.path.join(f"../test_results/result_cross/bdb2{domain_name.lower()}", x, f"sup_num_{shot}.json"), "r") as f:
        res = json.load(f)
    bdb[x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if not math.isnan(d):
            bdb[x].append(d)
            mean += d
        #bdb[x].append(mean / 10)
    chembl[x] = []
    with open(os.path.join(f"../test_results/result_cross/chembl2{domain_name.lower()}", x, f"sup_num_{shot}.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k] ])
        if not math.isnan(d):
            chembl[x].append(d)
            mean += d

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


ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"
plot_legend = False
if not plot_legend:
    ax = plot_settings.get_square_axis()
else:
    ax = plot_settings.get_wider_axis()
plot_utils.grouped_barplot(
        ax, means, 
        datasets,
        xlabel='', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=0, anchorpoint='center',
        legend_loc='upper left',
        min_val=min_val, scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if domain_name == "KIBA":
    if ylabel == "r2":
        ax.yaxis.set_major_locator(MultipleLocator(0.03))
        ax.set_ylim(0.00, 0.18)
    elif ylabel == "RMSE":
        ax.yaxis.set_major_locator(MultipleLocator(0.04))
        ax.set_ylim(0.72, 0.92)
else:
    if ylabel == "r2":
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(0.00, 0.30)
    elif ylabel == "RMSE":
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_ylim(0.90, 1.20)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()


if ylabel == "r2" and domain_name == "KIBA":
    plt.savefig(f'./figs/3.c.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "RMSE" and domain_name == "KIBA":
    plt.savefig(f'./figs/3.d.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "r2" and domain_name == "Davis":
    plt.savefig(f'./figs/3.e.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "RMSE" and domain_name == "Davis":
    plt.savefig(f'./figs/3.f.figure_cross_domain_{domain_name}_{ylabel}.pdf')


print("finish")