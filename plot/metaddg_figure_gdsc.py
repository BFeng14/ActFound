#!/usr/bin/env python
# coding: utf-8



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

models = ['no-pretraining', 'BDB-pretrain', 'ChEMBL-pretrain']
models_cvt = {'BDB-pretrain': 'onBDB',
              'ChEMBL-pretrain': 'onChEMBL',
              'no-pretraining': 'random-init',}

import os
import json
fsmol = [{}, {}, {}]
for x in models:
    dir = x.split("-")[0].lower()
    with open(os.path.join("/home/fengbin/meta_delta/test_result_gdsc",  dir, "sup_num_16.json"), "r") as f:
        res = json.load(f)
    fsmol[0][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data["r2"]) for data in res[k]])
        fsmol[0][x].append(d)
        mean += d

    fsmol[1][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data["rmse"]) for data in res[k]])
        fsmol[1][x].append(d)
        mean += d

    fsmol[2][x] = []
    for k in res:
        mean = 0
        d = np.mean([float(data["rho_s"]) for data in res[k]])
        fsmol[2][x].append(d)
        mean += d



colors = [plot_settings.get_model_colors(mod) for mod in models]


labels = [models_cvt.get(x, x) for x in models]
means_all = []
stderrs_all = []
for i in range(3):
    means = []
    stderrs = []
    for k in models:
        means.append(np.mean(fsmol[i][k]))
        stderrs.append(np.std(fsmol[i][k] / np.sqrt(len(fsmol[i][k]))))
    means_all.append(means)
    stderrs_all.append(stderrs)


import matplotlib.pyplot as plt

ylabel_all = ["r2", "RMSE", "$\\rho_S$"]
for i in range(3):
    plt.figure(figsize=(int(plot_settings.FIG_WIDTH), plot_settings.FIG_HEIGHT*0.4))
    ax_i = plt.subplot(1, 1, 1)
    min_val = np.min(np.array(means_all[i]) - np.array(stderrs_all[i]))
    min_val = 0. #max(min_val - 0.01, 0.)

    plot_utils.bar_ploth(
        ax_i, data=means_all[i], errs=stderrs_all[i], data_labels=labels,
        xlabel="", ylabel=ylabel_all[i] if ylabel_all[i] != "r2" else "r$^2$", rotangle=0, color=colors,
        min_val=min_val, invert_axes=False)
    if i == 0:
        ax_i.set_xlim(0.790, 0.770)
    elif i== 1:
        ax_i.set_xlim(0.650, 0.620)
    else:
        ax_i.set_xlim(0.630, 0.600)

    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    ax_i.xaxis.set_major_locator(MultipleLocator(0.01))
    ax_i.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  

    ax_i.spines['left'].set_visible(False)
    ax_i.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'./figs/4.f.figure_gdsc_{i}.pdf')


