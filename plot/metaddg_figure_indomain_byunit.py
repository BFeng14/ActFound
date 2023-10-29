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
assay_info_dict = json.load(open("../datas/assay_infos_chembl.json", "r"))

def is_does(unit):
    return unit in ['mg.kg-1', 'mg kg-1',
                       'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                       'M kg-1', 'pmg kg-1']

def is_mass(unit):
    return unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                       'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                       "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                       'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml"] and not is_does(unit)

result_dict = {}
for x in models:
    with open(os.path.join("../test_results/result_indomain/chembl", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        unit_type = "molar"
        unit = assay_info_dict[k]["unit"]
        if is_does(unit):
            unit_type = "dosage"
        elif is_mass(unit):
            unit_type = "density"

        if unit_type not in result_dict:
            result_dict[unit_type] = {}

        if x not in result_dict[unit_type]:
            result_dict[unit_type][x] = []
        result_dict[unit_type][x].append(d)
        mean += d

ax = plot_settings.get_wider_axis(double=False)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
mean = {}
std = {}

means_all = []
stderrs_all = []
unit_keys = list(result_dict.keys())
for unit in unit_keys:
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(result_dict[unit][k]))
        stderrs.append(np.std(result_dict[unit][k] / np.sqrt(len(result_dict[unit][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)

datasets = unit_keys

# import scipy
# from scipy import stats
# t_test = stats.ttest_rel(result_dict['density']['actfound_fusion'], result_dict['density']['maml'], alternative="greater")[1]
# print(t_test)

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
    # ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.yticks([0.15,0.25,0.35,0.45,0.55])
    ax.set_ylim(0.15, 0.55)
elif ylabel == "RMSE":
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  
    ax.set_ylim(0.45, 0.70)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plot_utils.format_ax(ax)
plt.title("Assay unit", size=18)
if plot_legend:
    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=2)
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01), prop={'size': 12})
plt.tight_layout()


plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/supplement.4.figure_chembl_indomain_{ylabel}_byunit.pdf')
print("finish", ylabel)



