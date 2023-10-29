#!/usr/bin/env python
# coding: utf-8

import warnings
import plot_utils
import plot_settings
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

sys.path.append(os.path.join(sys.path[0], '../'))
warnings.filterwarnings('ignore')

datasets = ["16-shot", "32-shot", "64-shot", "128-shot"]
models = ['actfound_fusion', 'actfound_transfer', 'ADKT-IFT', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']
models_cvt = plot_settings.models_cvt

metric_name = sys.argv[1]
if metric_name == "rmse":
    models = ['actfound_fusion', 'actfound_transfer', 'ADKT-IFT', 'maml', 'protonet', 'DKT', 'RF', 'GPST', 'KNN']
elif metric_name == "R2os":
    models = ['actfound_fusion', 'actfound_transfer', 'ADKT-IFT', 'protonet', 'DKT', 'RF', 'GPST']

fsmol = [{}, {}, {}, {}]
for x in models:
    if not os.path.exists(os.path.join("../test_results/result_indomain/fsmol", x)):
        continue
    with open(os.path.join("../test_results/result_indomain/fsmol", x, "sup_num_16.json"), "r") as f:
        try:
            res = json.load(f)
        except Exception as e:
            print(f)
            raise e
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d

    fsmol[1][x] = []
    with open(os.path.join("../test_results/result_indomain/fsmol", x, "sup_num_32.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[1][x].append(d)
        mean += d

    fsmol[2][x] = []
    with open(os.path.join("../test_results/result_indomain/fsmol", x, "sup_num_64.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[2][x].append(d)
        mean += d

    fsmol[3][x] = []
    with open(os.path.join("../test_results/result_indomain/fsmol", x, "sup_num_128.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[3][x].append(d)
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
# ax = plot_settings.get_wider_axis(double=True)
plt.figure(figsize=(int(plot_settings.FIG_WIDTH * 2.5), plot_settings.FIG_HEIGHT))
ax = plt.subplot(1, 1, 1)
plot_legend = True
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel='', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
    nested_color=colors, nested_errs=stderrs_all, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if ylabel == "r2":
    ax.yaxis.set_major_locator(MultipleLocator(0.1))  
    ax.set_ylim(0.00, 0.50)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.40, 0.65)
elif ylabel == "R2os":
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.set_ylim(0.00, 0.50)
    
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=4)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01), prop={'size': 12})
plt.title("FS-MOL", size=16+2)
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/2.c.figure_fsmol_indomain_{ylabel}.pdf')
elif ylabel == "RMSE":
    plt.savefig(f'./figs/supplement.1.figure_fsmol_indomain_{ylabel}.pdf')
elif ylabel == "R2os":
    plt.savefig(f'./figs/supplement.2.figure_fsmol_indomain_{ylabel}.pdf')
print("finish")