#!/usr/bin/env python
# coding: utf-8

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

sys.path.append(os.path.join(sys.path[0], '../'))
import plot_settings
import plot_utils
import warnings
warnings.filterwarnings('ignore')

models_cvt = plot_settings.models_cvt

metric_name = "r2"
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']

assay_info_dict = json.load(open("/home/fengbin/meta_delta/assay_infos.json", "r"))

BAO_format_dict = """BAO_0000218: organism-based assay
BAO_0000219: cell-based assay
BAO_0000357: single protein assay""".split("\n")
BAO_format_dict = [x.split(": ") for x in BAO_format_dict]
BAO_format_dict = {x[0]: x[1] for x in BAO_format_dict}

result_dict = {}
for x in models:
    with open(os.path.join("../test_results/result_indomain/chembl", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        bao_id = assay_info_dict[k]["bao_format"]
        if bao_id not in BAO_format_dict.keys():
            continue
        if bao_id not in result_dict:
            result_dict[bao_id] = {}

        if x not in result_dict[bao_id]:
            result_dict[bao_id][x] = []
        result_dict[bao_id][x].append(d)
        mean += d

ax = plot_settings.get_wider_axis(double=False)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
mean = {}
std = {}

means_all = []
stderrs_all = []
BAO_keys = list(result_dict.keys())
for bao_id in BAO_keys:
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(result_dict[bao_id][k]))
        stderrs.append(np.std(result_dict[bao_id][k] / np.sqrt(len(result_dict[bao_id][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)

datasets = [BAO_format_dict[x] for x in BAO_keys]

# In[5]:

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
    min_val=0., scale=2)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
if ylabel == "r2":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(0.15, 0.50)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(0.40, 0.70)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plot_utils.format_ax(ax)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right',
                                ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/supplement.3.figure_indomain_{ylabel}_byBAO.pdf')
    print("finish", ylabel)



