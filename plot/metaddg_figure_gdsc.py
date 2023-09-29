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

models = ['BDB-pretrain', 'ChEMBL-pretrain', 'no-pretrain']
models_cvt = {'Meta-DDG': 'Meta-Delta',
              'maml': 'MAML',
              'transfer_delta': 'Transfer-Delta',
              'transfer_qsar': 'Transfer-QSAR'}

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
        d = np.mean([float(data["R2os"]) for data in res[k]])
        fsmol[2][x].append(d)
        mean += d

# In[4]:
ax = plot_settings.get_wider_axis(double=True)
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
plt.figure(figsize=(int(plot_settings.FIG_WIDTH * (3/2)), plot_settings.FIG_HEIGHT))
ylabel_all = ["r2", "RMSE", "R2os"]
for i in range(3):
    ax_i = plt.subplot(1, 3, i+1)
    min_val = np.min(np.array(means_all[0]) - np.array(stderrs_all[0]))
    min_val = max(min_val-0.1, 0.)
    plot_utils.bar_plot(
        ax_i, data=means_all[0], errs=stderrs_all[0], data_labels=labels,
        xlabel="", ylabel=ylabel_all[i], rotangle=0,
        min_val=min_val)


plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', ncols=2)
plot_utils.put_legend_outside_plot(ax, anchorage=(1.01, 1.01))
plt.tight_layout()

plt.show()
plt.savefig(f'./figs/figure_gdsc.pdf')

