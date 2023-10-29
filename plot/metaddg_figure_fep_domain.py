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

datasets = ["20%", "40%", "60%", "80%"]
models = ['actfound_fusion', 'actfound_transfer', 'maml', 'protonet', 'transfer_qsar']
models_cvt = plot_settings.models_cvt

metric_name = sys.argv[1]
domain_name = sys.argv[2]
dataset_name = sys.argv[3]

fepset = [{}, {}, {}, {}]
for x in models:
    with open(os.path.join(f"../test_results/result_fep_new/{domain_name}/{dataset_name}", x, "sup_num_0.2.json"), "r") as f:
        res = json.load(f)
    fepset[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fepset[0][x].append(d)
        mean += d

    fepset[1][x] = []
    with open(os.path.join(f"../test_results/result_fep_new/{domain_name}/{dataset_name}", x, "sup_num_0.4.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fepset[1][x].append(d)
        mean += d

    fepset[2][x] = []
    with open(os.path.join(f"../test_results/result_fep_new/{domain_name}/{dataset_name}", x, "sup_num_0.6.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fepset[2][x].append(d)
        mean += d

    fepset[3][x] = []
    with open(os.path.join(f"../test_results/result_fep_new/{domain_name}/{dataset_name}", x, "sup_num_0.8.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fepset[3][x].append(d)
        mean += d

colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(4):
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(fepset[i][k]))
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
if plot_legend:
    plt.figure(figsize=(int(plot_settings.FIG_WIDTH * 2.5), plot_settings.FIG_HEIGHT))
    ax = plt.subplot(1, 1, 1)
else:
    ax = plot_settings.get_wider_axis(double=True)
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel='Percentage of the fine-tuning data', ylabel=ylabel if ylabel != "r2" else "r$^2$", color_legend=labels if plot_legend else None,
    nested_color=colors, nested_errs=stderrs_all, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

plot_utils.format_ax(ax)
if plot_legend:
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper right', scatterpoints=1, ncol=1, bbox_to_anchor=(1.35, 1.01),
    #            markerscale=20)
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='center right', ncols=4)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 0.5))

if metric_name == "r2":
    plt.axhline(y=0.569, color='#b2182b', linestyle='--', lw=1)
elif metric_name == "rmse":
    plt.axhline(y=0.591, color='#b2182b', linestyle='--', lw=1)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

if dataset_name=="chembl":
    if metric_name == "r2":
        if domain_name == "fep":
            ax.set_yticks([0.15 + 0.1 * x for x in range(6 + 1)])
            ax.set_ylim(0.15, 0.75)
        elif domain_name == "fep_opls4":
            ax.set_yticks([0.15 + 0.1 * x for x in range(5 + 1)])
            ax.set_ylim(0.15, 0.65)
    elif metric_name == "rmse":
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_ylim(0.40, 1.0)
else:
    if metric_name == "r2":
        if domain_name == "fep":
            ax.set_yticks([0.15+0.1*x for x in range(6+1)])
            ax.set_ylim(0.15, 0.75)
        elif domain_name == "fep_opls4":
            ax.set_yticks([0.15+0.1*x for x in range(5+1)])
            ax.set_ylim(0.15, 0.65)
    elif metric_name == "rmse":
        if domain_name == "fep_opls4":
            ax.yaxis.set_major_locator(MultipleLocator(0.10))
            # plt.yticks([0.35, 0.50, 0.65, 0.80, 1.0])
            ax.set_ylim(0.40, 1.0)
        elif domain_name == "fep":
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            # plt.yticks([0.25, 0.40, 0.55, 0.70, 0.85, 1.00])
            ax.set_ylim(0.20, 1.00)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tight_layout()
# plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=1)
# plt.legend(loc='upper right')
# plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))


if dataset_name == "bdb":
    if metric_name == "r2" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.11.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.12.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "r2" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/supplement.13.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/supplement.14.{metric_name}_{domain_name}_{dataset_name}.pdf')
else:
    if metric_name == "r2" and domain_name == "fep":
        plt.savefig(f'./figs/4.a.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep":
        plt.savefig(f'./figs/4.b.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "r2" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/4.c.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/4.d.{metric_name}_{domain_name}_{dataset_name}.pdf')
plt.show()

print("finish")