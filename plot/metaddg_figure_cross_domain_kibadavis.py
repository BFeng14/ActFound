#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
try:
    shot = int(sys.argv[3])
except:
    shot = 16

datasets = [f"ChEMBL->{domain_name}", f"BDB->{domain_name}"]
models = ['meta_delta_fusion', 'transfer_delta', 'maml', 'protonet', 'DKT', 'CNP', 'transfer_qsar']
models_cvt = {'meta_delta_fusion': 'MetaLigand',
              'maml': 'MAML',
              'transfer_delta': 'TransferLigand',
              'transfer_qsar': 'TransferQSAR',
              'protonet': 'ProtoNet'}

if metric_name == "rmse":
    models = ['meta_delta_fusion', 'transfer_delta', 'maml', 'protonet', 'DKT']

import os
import json
import math
bdb = {}
chembl = {}
for x in models:
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_cross/bdb2{domain_name.lower()}", x, f"sup_num_{shot}.json"), "r") as f:
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
    with open(os.path.join(f"/home/fengbin/meta_delta_master/result_cross/chembl2{domain_name.lower()}", x, f"sup_num_{shot}.json"), "r") as f:
        res = json.load(f)
    for k in res:
        mean = 0
        d = np.mean([float(data[metric_name]) for data in res[k] ])
        if not math.isnan(d):
            chembl[x].append(d)
            mean += d

# for x in bdb:
#     print(len(bdb[x]))
# for x in chembl:
#     print(len(chembl[x]))


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
min_val = max(min_val-(max_val-min_val)*0.15, 0.)

ylabel = metric_name
if metric_name == "rmse":
    ylabel = "RMSE"
ax = plot_settings.get_wider_axis(double=False)
plot_utils.grouped_barplot(
        ax, means, 
        datasets,
        xlabel='', ylabel=ylabel, color_legend=None,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=0, anchorpoint='center',
        legend_loc='upper left',
        min_val=min_val, scale=2)
    
plot_utils.format_ax(ax)
plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=2)
plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

if shot > 16:
    if shot == 32:
        plt.savefig(f'./figs/supplement.5.figure_cross_domain_{domain_name}_{ylabel}.pdf')
    elif shot == 64:
        plt.savefig(f'./figs/supplement.6.figure_cross_domain_{domain_name}_{ylabel}.pdf')
    elif shot == 128:
        plt.savefig(f'./figs/supplement.7.figure_cross_domain_{domain_name}_{ylabel}.pdf')

if ylabel == "r2" and domain_name == "KIBA":
    plt.savefig(f'./figs/3.c.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "RMSE" and domain_name == "KIBA":
    plt.savefig(f'./figs/3.d.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "r2" and domain_name == "Davis":
    plt.savefig(f'./figs/3.e.figure_cross_domain_{domain_name}_{ylabel}.pdf')
elif ylabel == "RMSE" and domain_name == "Davis":
    plt.savefig(f'./figs/3.f.figure_cross_domain_{domain_name}_{ylabel}.pdf')


