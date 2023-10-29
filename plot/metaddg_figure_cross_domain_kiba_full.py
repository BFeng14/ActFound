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
import math

sys.path.append(os.path.join(sys.path[0], '../'))
warnings.filterwarnings('ignore')

train_data = sys.argv[1]
if train_data == "chembl":
    test_name = "ChEMBL to KIBA"
elif train_data == "bdb":
    test_name = "BindingDB to KIBA"
datasets = [f"16-shot", f"32-shot", f"64-shot", f"128-shot"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar']
models_cvt = plot_settings.models_cvt
metric_name = "r2"

kiba = [{}, {}, {}, {}]
for x in models:
    if not os.path.exists(os.path.join(f"../test_results/result_cross/{train_data}2kiba", x)):
        continue
    with open(os.path.join(f"../test_results/result_cross/{train_data}2kiba", x, "sup_num_16.json"), "r") as f:
        try:
            res = json.load(f)
        except Exception as e:
            print(f)
            raise e
    kiba[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if not math.isnan(d):
            kiba[0][x].append(d)
            mean += d

    kiba[1][x] = []
    with open(os.path.join(f"../test_results/result_cross/{train_data}2kiba", x, "sup_num_32.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if not math.isnan(d):
            kiba[1][x].append(d)
            mean += d

    kiba[2][x] = []
    with open(os.path.join(f"../test_results/result_cross/{train_data}2kiba", x, "sup_num_64.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if not math.isnan(d):
            kiba[2][x].append(d)
            mean += d

    kiba[3][x] = []
    with open(os.path.join(f"../test_results/result_cross/{train_data}2kiba", x, "sup_num_128.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        if not math.isnan(d):
            kiba[3][x].append(d)
            mean += d


ax = plot_settings.get_wider_axis(double=True)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(4):
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(kiba[i][k]))
        stderrs.append(np.std(kiba[i][k] / np.sqrt(len(kiba[i][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)

min_val = np.min(np.array(means_all) - np.array(stderrs_all))
max_val = np.max(np.array(means_all) + np.array(stderrs_all))
min_val = 0.0 #max(min_val-(max_val-min_val)*0.15, 0.)

ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"
# ax = plot_settings.get_wider_axis(double=True)
plt.figure(figsize=(int(plot_settings.FIG_WIDTH * 2.5), plot_settings.FIG_HEIGHT))
ax = plt.subplot(1, 1, 1)
plot_legend = True
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel=f'Number of the fine-tuning data', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
    nested_color=colors, nested_errs=stderrs_all, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if ylabel == "r2":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.00, 0.25)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right',
                             ncols=4)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01), prop={'size': 12})
plt.title(test_name, size=18)
plt.tight_layout()

plt.show()
if train_data == "chembl":
    plt.savefig(f'./figs/supplement.5.figure_cross_domain_{train_data}toKIBA_16to128_{ylabel}.pdf')
else:
    plt.savefig(f'./figs/supplement.6.figure_cross_domain_{train_data}toKIBA_16to128_{ylabel}.pdf')
