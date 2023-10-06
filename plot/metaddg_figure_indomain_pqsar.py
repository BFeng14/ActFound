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

datasets = ["pQSAR-ChEMBL"]
models = ['meta_delta_fusion', 'transfer_delta', 'maml', 'DKT', 'protonet', 'CNP', 'transfer_qsar', 'RF', 'GPST', 'KNN']
models_cvt = {'meta_delta_fusion': 'MetaLigand',
              'maml': 'MAML',
              'transfer_delta': 'TransferLigand',
              'transfer_qsar': 'TransferQSAR',
              'protonet': 'ProtoNet'}

# In[3]:
metric_name = sys.argv[1]
if metric_name == "rmse":
    models = ['meta_delta_fusion', 'transfer_delta', 'maml', 'DKT', 'protonet', 'RF', 'GPST', 'KNN']
elif metric_name == "R2os":
    models = ['meta_delta_fusion', 'transfer_delta', 'DKT', 'protonet', 'RF', 'GPST']

fsmol = [{}]
for x in models:
    if not os.path.exists(os.path.join("/home/fengbin/meta_delta_master/result_indomain/fsmol", x)):
        continue
    with open(os.path.join("/home/fengbin/meta_delta_master/result_indomain/pqsar", x, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d


# In[4]:
ax = plot_settings.get_wider_axis(double=True)
colors = [plot_settings.get_model_colors(mod) for mod in models]
labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(1):
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
ax = plot_settings.get_wider_axis(double=False)
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

plt.show()
if ylabel == "r2":
    plt.savefig(f'./figs/2.d.figure_pqsar_indomain_{ylabel}.pdf')
else:
    if ylabel == "RMSE":
        plt.savefig(f'./figs/supplement.3.figure_pqsar_indomain_{ylabel}.pdf')