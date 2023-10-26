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

root_path_list = []
root_path_list.append("./test_results/result_ood")
root = "./test_results/result_indomain"
root_path_list += [os.path.join(root, x) for x in os.listdir(root)]
root = "./test_results/result_cross"
root_path_list += [os.path.join(root, x) for x in os.listdir(root)]
root = "./test_results/result_fep_new/fep"
root_path_list += [os.path.join(root, x) for x in os.listdir(root)]
root = "./test_results/result_fep_new/fep_opls4"
root_path_list += [os.path.join(root, x) for x in os.listdir(root)]

sup_num_list = ["6", "10", "16", "32", "64", "128", "0.2", "0.4", "0.6", "0.8"]
for root_path in root_path_list:
    for sup_num in sup_num_list:
        path_meta_delta = f"{root_path}/actfound"
        path_transfer_delta = f"{root_path}/actfound_transfer"
        path_fusion = f"{root_path}/actfound_fusion"
        if not os.path.exists(f"{path_meta_delta}/sup_num_{sup_num}.json"):
            continue
        if not os.path.exists(path_fusion):
            os.mkdir(path_fusion)
        result_meta_delta = json.load(open(f"{path_meta_delta}/sup_num_{sup_num}.json", "r"))
        result_transfer_delta = json.load(open(f"{path_transfer_delta}/sup_num_{sup_num}.json", "r"))

        result_meta_fusion = {}
        for assay_id in result_transfer_delta.keys():
            ax = result_meta_delta[assay_id]
            bx = result_transfer_delta[assay_id]
            result_assay = []
            for a, b in zip(ax, bx):
                a_loss = a['each_step_loss'][0]
                b_loss = b['each_step_loss'][0]

                result_a = a['pred']
                result_b = b['pred']
                result_true = a['ture']
                if a_loss >= b_loss-0.1:
                    fusion_pred = [(x*0.5+y*0.5) for x, y in zip(result_a, result_b)]
                    y_train_mean = a['y_train_mean']
                    rmse, r2, R2os = get_metric(result_true, fusion_pred, y_train_mean)
                    fusion_info = copy.deepcopy(a)
                    fusion_info['pred'] = fusion_pred
                    fusion_info['rmse'] = rmse
                    fusion_info['r2'] = r2
                    fusion_info['R2os'] = R2os
                    result_assay.append(fusion_info)
                else:
                    result_assay.append(a)
            result_meta_fusion[assay_id] = result_assay
        print(get_mean_r2(result_meta_delta), get_mean_r2(result_meta_fusion))
        print("writing to ", f"{path_fusion}/sup_num_{sup_num}.json \n")
        json.dump(result_meta_fusion, open(f"{path_fusion}/sup_num_{sup_num}.json", "w"))

