import copy
import json
import numpy as np
import os
import math

def get_mean_r2(res_dict):
    r2_all = []
    for task_name, task_test_results in res_dict.items():
        r2_all.append(np.mean([eval_res["r2"] for eval_res in task_test_results]))
    return np.mean(r2_all)

def get_metric(y_true, y_pred, y_train_mean):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    r2 = np.corrcoef(y_true, y_pred)[0, 1]
    if math.isnan(r2) or r2 < 0.:
        r2 = 0.
    else:
        r2 = r2 ** 2

    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - y_train_mean) ** 2).sum(axis=0, dtype=np.float64)
    if denominator == 0:
        R2os = 0.0
    else:
        R2os = 1.0 - (numerator / denominator)

    return rmse, r2, R2os

root_path = "./test_results/result_ood_tmp"
model_name_list = ["meta_delta", "transfer_delta"]
for model_name in model_name_list:
    path_normal = f"{root_path}/normal/{model_name}"
    path_inverse = f"{root_path}/inverse/{model_name}"
    path_fusion = f"./test_results/result_ood/{model_name}"
    if not os.path.exists(path_fusion):
        os.mkdir(path_fusion)
    result_normal = json.load(open(f"{path_normal}/sup_num_16.json", "r"))
    result_inverse = json.load(open(f"{path_inverse}/sup_num_16.json", "r"))

    result_meta_fusion = {}
    for assay_id in result_inverse.keys():
        ax = result_normal[assay_id]
        bx = result_inverse[assay_id]
        result_assay = []
        for a, b in zip(ax, bx):
            a_loss = a['each_step_loss'][0]
            b_loss = b['each_step_loss'][0]

            if a_loss >= b_loss:
                result_assay.append(b)
            else:
                result_assay.append(a)
        result_meta_fusion[assay_id] = result_assay
    print(get_mean_r2(result_normal), get_mean_r2(result_meta_fusion))
    print("writing to ", f"{path_fusion}/sup_num_16.json \n")
    json.dump(result_meta_fusion, open(f"{path_fusion}/sup_num_16.json", "w"))
