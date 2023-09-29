#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

datasets = ["16-shot", "32-shot", "64-shot", "128-shot"]
models = ['meta_delta_fusion', 'transfer_delta', 'ADKT-IFT', 'maml', 'DKT', 'ProtoNet', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']
models_cvt = {'meta_delta_fusion': 'Meta-Delta',
              'maml': 'MAML',
              'transfer_delta': 'Transfer-Delta',
              'transfer_qsar': 'Transfer-QSAR'}

# In[3]:
metric_name = sys.argv[1]
if metric_name == "rmse":
    models = ['meta_delta_fusion', 'transfer_delta', 'ADKT-IFT', 'maml', 'DKT', 'ProtoNet', 'RF', 'GPST', 'KNN']
elif metric_name == "R2os":
    models = ['meta_delta_fusion', 'transfer_delta', 'ADKT-IFT', 'DKT', 'ProtoNet', 'RF', 'GPST']

fsmol = [{}, {}, {}, {}]
for x in models:
    if not os.path.exists(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x)):
        continue
    with open(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d

    fsmol[1][x] = []
    with open(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x, "sup_num_32.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[1][x].append(d)
        mean += d

    fsmol[2][x] = []
    with open(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x, "sup_num_64.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[2][x].append(d)
        mean += d

    fsmol[3][x] = []
    with open(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x, "sup_num_128.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[3][x].append(d)
        mean += d

# In[4]:
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
ax = plot_settings.get_wider_axis(double=True)
plot_utils.grouped_barplot(
    ax, means_all,
    datasets,
    xlabel='', ylabel=ylabel, color_legend=labels,
    nested_color=colors, nested_errs=stderrs_all, tickloc_top=False, rotangle=0, anchorpoint='center',
    legend_loc='upper left',
    min_val=min_val, scale=2)

plot_utils.format_ax(ax)
plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right',
                         ncols=4)
plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/2.c.figure_fsmol_indomain_{ylabel}.pdf')
else:
    if ylabel == "RMSE":
        plt.savefig(f'./figs/supplement.1.figure_fsmol_indomain_{ylabel}.pdf')
    elif ylabel == "R2os":
        plt.savefig(f'./figs/supplement.2.figure_fsmol_indomain_{ylabel}.pdf')
