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

datasets = ["0.2-shot", "0.4-shot", "0.6-shot", "0.8-shot"]
models = ['meta_delta_fusion', 'transfer_delta', 'maml', 'protonet', 'transfer_qsar']
models_cvt = {'meta_delta_fusion': 'MetaLigand',
              'maml': 'MAML',
              'transfer_delta': 'TransferLigand',
              'transfer_qsar': 'TransferQSAR',
              'protonet': 'ProtoNet'}

metric_name = sys.argv[1]
domain_name = sys.argv[2]
dataset_name = sys.argv[3]

fsmol = [{}, {}, {}, {}]
for x in models:
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_fep/{domain_name}/{dataset_name}", x, "sup_num_0.2.json"), "r") as f:
        res = json.load(f)
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d

    fsmol[1][x] = []
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_fep/{domain_name}/{dataset_name}", x, "sup_num_0.4.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[1][x].append(d)
        mean += d

    fsmol[2][x] = []
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_fep/{domain_name}/{dataset_name}", x, "sup_num_0.6.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[2][x].append(d)
        mean += d

    fsmol[3][x] = []
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_fep/{domain_name}/{dataset_name}", x, "sup_num_0.8.json"), "r") as f:
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
plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=4)
plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()
if metric_name == "r2":
    plt.axhline(y=0.575, color='r', linestyle='solid')
elif metric_name == "rmse":
    plt.axhline(y=0.602*1.38, color='r', linestyle='solid')


if dataset_name == "bdb":
    if metric_name == "r2" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.8.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.9.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "r2" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/supplement.10.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep_opls4":
        plt.savefig(f'./figs/supplement.11.{metric_name}_{domain_name}_{dataset_name}.pdf')
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
