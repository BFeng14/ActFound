
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
import plot_settings
import plot_utils
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

data_name = sys.argv[1]
try:
    train_name = sys.argv[2]
except:
    train_name = "chembl"

res_dict = json.load(open(f"../test_results/result_indomain/{train_name}/meta_delta_fusion/sup_num_16.json", "r"))
res_dict_1 = json.load(open(f"../test_results/result_cross/{train_name}2davis/meta_delta_fusion/sup_num_16.json", "r"))
res_dict_2 = json.load(open(f"../test_results/result_cross/{train_name}2kiba/meta_delta_fusion/sup_num_16.json", "r"))

r2 = []
v = []
for task_name, task_test_results in res_dict.items():
    # for eval_res in task_test_results:
    r2.append(np.mean([eval_res["r2"] for eval_res in task_test_results]))
    v.append(np.mean([eval_res["each_step_loss"][0] for eval_res in task_test_results]))

if data_name == "Davis":
    for task_name, task_test_results in res_dict_1.items():
        # for eval_res in task_test_results:
        r2.append(np.mean([eval_res["r2"] for eval_res in task_test_results]))
        v.append(np.mean([eval_res["each_step_loss"][0] for eval_res in task_test_results]))
else:
    for task_name, task_test_results in res_dict_2.items():
        # for eval_res in task_test_results:
        r2.append(np.mean([eval_res["r2"] for eval_res in task_test_results]))
        v.append(np.mean([eval_res["each_step_loss"][0] for eval_res in task_test_results]))

print(np.corrcoef(r2, v)[0, 1])
plt.figure(figsize=(4, 4))
ax = plt.subplot(1, 1, 1)
style = MarkerStyle('o')

color_indomain = "#8c510a"
color_cross = "#01665e"
if train_name == "chembl":
    in_domain_label = "In-domain"
else:
    in_domain_label = "In-domain"
ax.scatter(v[:len(res_dict)], r2[:len(res_dict)], alpha=1.0, s=14, marker=style, c=color_indomain, label=in_domain_label)
ax.scatter(v[len(res_dict):], r2[len(res_dict):], alpha=1.0, s=14, marker=style, c=color_cross, label=data_name)

if data_name == "KIBA":
    ax.set_xlabel("First step loss on KIBA", size=14)
else:
    ax.set_xlabel("First step loss on Davis", size=14)

ax.set_ylabel("r$^2$", size=16)
#setting the Pearson
r = np.corrcoef(r2, v)[0, 1]
ax = plt.gca() # Get a matplotlib's axes instance
#setting theline
b, a = np.polyfit(v,r2, deg=1)
xseq = np.linspace(0, 10, num=100)
ax.plot(xseq, a + b * xseq, color="gray", lw=0.75, linestyle="--")
plt.text(0.06, 0.1, "$\\rho_P$ ={:.2f}".format(r), transform=ax.transAxes, s=14)
plt.xticks(size=16)
plt.yticks(size=16)

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_ylim(0.00, 1.00)
ax.xaxis.set_major_locator(MultipleLocator(0.3))
ax.set_xlim(0.00, 1.5)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  
plot_utils.format_ax(ax)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper center', scatterpoints=1, ncol=2, bbox_to_anchor=(0.5, 1.15), markerscale=2.0, frameon=False, prop={'size': 14})
# plot_utils.put_legend_outside_plot(ax, anchorage=(0.5, 1.01))
plt.tight_layout()

if train_name == "chembl":
    if data_name == "KIBA":
        plt.savefig('./figs/3.g1.first_step_loss_kiba-domain.pdf')
    else:
        plt.savefig('./figs/3.g2.first_step_loss_davis-domain.pdf')
else:
    if data_name == "KIBA":
        plt.savefig('./figs/supplement.7.first_step_loss_kiba-domain_bdb.pdf')
    else:
        plt.savefig('./figs/supplement.8.first_step_loss_davis-domain_bdb.pdf')
plt.show()