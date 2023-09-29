#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import warnings
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
sys.path.append(os.path.join(sys.path[0], '../'))
warnings.filterwarnings('ignore')


metric_name = sys.argv[1]
domain_name = sys.argv[2]
dataset_name = sys.argv[3]
if domain_name == 'fep':
    root_path = f"/home/fengbin/meta_delta_master/result_fep/fep/{dataset_name}"
if domain_name == 'opls4':
    root_path = f"/home/fengbin/meta_delta_master/result_fep/fep_opls4/{dataset_name}"
models = ['meta_delta_fusion', 'maml',
          'protonet', 'transfer_delta', 'transfer_qsar']
models_cvt = {'meta_delta_fusion': 'Meta-Delta',
              'maml': 'MAML',
              'transfer_delta': 'Transfer-Delta',
              'transfer_qsar': 'Transfer-QSAR',
              'protonet': 'ProtoNet'}


def read_data(model_name):
    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y = []
    err = []
    for i in x:
        file_name = f"{root_path}/{model_name}/sup_num_{i}.json"
        res_dict = json.load(open(file_name, "r"))
        res_list = []
        for task_name, task_test_results in res_dict.items():
            mean_res = np.mean([eval_res[metric_name]
                               for eval_res in task_test_results])
            res_list.append(mean_res)
        y.append(np.mean(res_list))
        err.append(np.std(res_list / np.sqrt(len(res_list))))
    return x, y, np.array(err)


def get_final_data(models):
    all_data = {}
    for model_name in models:
        all_data[model_name] = {}
        x, y, err = read_data(model_name)
        all_data[model_name]['x'] = x
        all_data[model_name]['y'] = y
        all_data[model_name]['err'] = err
    final_data = {models_cvt.get(k): v for k, v in all_data.items()}
    return final_data


final_data = get_final_data(models)
color_set = sns.color_palette("Set1").as_hex()[:len(final_data)]


labels = []
sns.set_palette("Set1")
plt.figure(figsize=(10, 7))
counter = 0
for model_name, v in final_data.items():
    labels.append(model_name)
    x = final_data[model_name]['x']
    y = final_data[model_name]['y']
    err = final_data[model_name]['err']
    sns.lineplot(x, y, linewidth=3, markers=100)
    plt.fill_between(x, y+err, y-err, alpha=0.09,
                     color=color_set[counter], zorder=100,)
    counter += 1
if metric_name == "r2":
    if domain_name == 'fep':
        plt.ylim(0.2, 0.7, 0.5)
        plt.yticks()
        plt.title(f"{dataset_name.upper()} FEP test", size=20)

    if domain_name == 'opls4':
        plt.ylim(0.15, 0.65)
        plt.yticks()
        plt.title(f"{dataset_name.upper()} OPLS4 as support test", size=20)
    plt.xticks(ticks=x)
    plt.ylabel(metric_name, size=16)
    plt.axhline(y=0.575, color='k', linestyle='solid')
    labels.append('OPLS4')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=False, ncol=6, labels=labels)

if metric_name == "rmse":
    if domain_name == 'fep':
        # plt.ylim(0.5, 0.9, 0.5)
        # plt.yticks(np.arange(0.4, 0.9, step=0.5))
        plt.ylim(0.4, 0.95)
        plt.title(f"{dataset_name.upper()} FEP test", size=20)

    if domain_name == 'opls4':
        plt.ylim(0.5, 1.15)
        plt.title(f"{dataset_name.upper()} OPLS4 as support test", size=20)
    plt.xticks(ticks=x)
    plt.ylabel('RMSE', size=16)
    plt.axhline(y=0.602, color='k', linestyle='solid')
    labels.append('OPLS4')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=False, ncol=6, labels=labels)


if dataset_name == "bdb":
    if metric_name == "r2" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.8.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep":
        plt.savefig(f'./figs/supplement.9.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "r2" and domain_name == "opls4":
        plt.savefig(f'./figs/supplement.10.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "opls4":
        plt.savefig(f'./figs/supplement.11.{metric_name}_{domain_name}_{dataset_name}.pdf')
else:
    if metric_name == "r2" and domain_name == "fep":
        plt.savefig(f'./figs/4.a.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "fep":
        plt.savefig(f'./figs/4.b.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "r2" and domain_name == "opls4":
        plt.savefig(f'./figs/4.c.{metric_name}_{domain_name}_{dataset_name}.pdf')
    elif metric_name == "rmse" and domain_name == "opls4":
        plt.savefig(f'./figs/4.d.{metric_name}_{domain_name}_{dataset_name}.pdf')
plt.show()
