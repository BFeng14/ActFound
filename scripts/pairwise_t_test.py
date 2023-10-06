import json
import numpy as np
import scipy
from scipy import stats

kiba_outs = json.load(open("/home/fengbin/meta_delta_master/result_indomain/chembl/meta_delta_fusion/sup_num_16.json"))
kiba_next = json.load(open("/home/fengbin/meta_delta_master/result_indomain/chembl/transfer_delta/sup_num_16.json"))

best_all = []
a_r2 = []
b_r2 = []
r2_diff_all = []
zero_loss_diff_all = []
for k in kiba_next.keys():
    ax = kiba_outs[k]
    bx = kiba_next[k]
    a_r2.append(np.mean([x["rmse"] for x in ax]))
    b_r2.append(np.mean([x["rmse"] for x in bx]))

print(stats.ttest_rel(a_r2, b_r2))