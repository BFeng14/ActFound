import numpy as np
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
import gzip

def read_BDB_per_assay():
    data_dir = "/home/fengbin/datas/BDB/polymer"
    assays = []
    ligand_sets = {}
    split_cnt = 0

    for target_name in tqdm(list(os.listdir(data_dir))):
        for assay_file in os.listdir(os.path.join(data_dir, target_name)):
            assay_name = target_name + "/" + assay_file
            entry_assay = "_".join(assay_file.split("_")[:2])
            affi_idx = int(assay_file[-5])
            ligands = []
            affis = []
            file_lines = list(open(os.path.join(data_dir, target_name, assay_file), "r").readlines())
            for i, line in enumerate(file_lines):
                line = line.strip().split("\t")
                affi_prefix = ""
                pic50_exp = line[8+affi_idx].strip()
                if pic50_exp.startswith(">") or pic50_exp.startswith("<"):
                    continue
                    # affi_prefix = pic50_exp[0]
                    # pic50_exp = pic50_exp[1:]
                try:
                    pic50_exp = -math.log10(float(pic50_exp))
                except:
                    print("error ic50s:", pic50_exp)
                    continue
                smiles = line[1]
                affis.append(pic50_exp)
                ligand_info = {
                    "affi_idx": affi_idx,
                    "affi_prefix": affi_prefix,
                    "smiles": smiles,
                    "pic50_exp": pic50_exp,
                    "domain": "bdb"
                }
                ligands.append(ligand_info)
            
            pic50_exp_list = [x["pic50_exp"] for x in ligands]
            pic50_std = np.std(pic50_exp_list)
            if pic50_std <= 0.2:
                continue
            if len(ligands) < 20:
                continue
            ligand_sets[assay_name] = ligands

    print("split_cnt:", split_cnt)
    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


def read_gdsc():
    data_file = "/home/fengbin/datas/gdsc/data_dict.pkl"
    ligand_sets = pickle.load(open(data_file, "rb"))
    ligand_sets_new = {}
    for k, v in ligand_sets.items():
        ligand_sets_new[k] = v
        for ligand_info in v:
            ligand_info['pic50_exp'] = -math.log10(math.exp(ligand_info['pic50_exp']))
        # if len(ligand_sets_new) > 200:
        #     break
    return {"ligand_sets": ligand_sets_new,
            "assays": list(ligand_sets_new.keys())}


def read_BDB_merck():
    datas = json.load(open("/home/fengbin/datas/FEP/fep_data_final.json", "r"))
    ligand_sets = {}
    task2opls4 = {}
    pic50s_all = []
    for k, v in datas.items():
        ligands = []
        opls4_res = []
        errors = []
        for ligand_info in v:
            pic50_exp = -float(ligand_info["exp_dg"])
            opls4 = -float(ligand_info["pred_dg"])
            errors.append(pic50_exp - opls4)
            opls4_res.append(opls4)
            smiles = ligand_info["smiles"]
            ligands.append({
                "affi_prefix": "",
                "smiles": smiles,
                "pic50_exp": pic50_exp,
                "domain": "fep"
            })
            pic50s_all.append(pic50_exp)
        ligand_sets[k] = ligands
        task2opls4[k] = np.array(opls4_res)
    print(np.mean(pic50s_all))
    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}, task2opls4


def read_kiba():
    ligand_list = []
    ligands_dict = json.load(open("/home/fengbin/datas/DeepDTA/data/kiba/ligands_can.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    Y = pickle.load(open("/home/fengbin/datas/DeepDTA/data/kiba/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}
    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        ligands = []
        pic50s = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                pic50s.append((affi - 11.72) + 6.75)
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": (affi - 11.72) + 6.75
                    }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        stds.append(np.std(pic50s))
        ligand_sets[f"kiba_{assay_idx}"] = ligands
    print("stds", np.mean(stds))


    assay_id_dicts_new = {}
    for assay_id, ligands in ligand_sets.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 50:
            continue
        assay_id_dicts_new[assay_id] = ligands
    return {"ligand_sets": assay_id_dicts_new, "assays": list(assay_id_dicts_new.keys())}


def read_davis():
    ligand_list = []
    ligands_dict = json.load(open("/home/fengbin/datas/DeepDTA/data/davis/ligands_can.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    Y = pickle.load(open("/home/fengbin/datas/DeepDTA/data/davis/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}

    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        ligands = []
        pic50s = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                pic50s.append(-math.log10(affi))
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": -math.log10(affi)
                    }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        stds.append(np.std(pic50s))
        ligand_sets[f"davis_{assay_idx}"] = ligands

    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def read_fsmol_assay(split = "train", train_phase=1):
    cache_file = f"/home/fengbin/datas/fsmol/{split}_cache.pkl"
    if os.path.exists(cache_file):
        datas = pickle.load(open(cache_file, "rb"))
        for k, v in datas["ligand_sets"].items():
            ligands_new = []
            for item in v:
                try:
                    ligands_new.append({
                        "smiles": item["SMILES"],
                        "pic50_exp": eval(item["LogRegressionProperty"]),
                        "domain": "fsmol"
                    })
                except:
                    pass
            datas["ligand_sets"][k] = ligands_new
        return datas


    # if train_phase == 0 and split == "train":
    #     return {"ligand_sets": {}, "assays": []}
    # fsmol_path = f"/home/fengbin/meta_delta/fsmol_data/{split}"
    # ligand_sets = {}
    # if split == "test":
    #     test_file = open("/home/fengbin/meta_delta/fsmol_data/regression_test_adkfift.csv", "r").readlines()
    #     split_data = [x.split(",")[0] for x in test_file][1:]
    # else:
    #     split_path = json.load(open("/home/fengbin/meta_delta/fsmol_data/fsmol-0.1.json", "r"))
    #     split_data = split_path[split]
    # for file in tqdm(os.listdir(fsmol_path)):
    #     assay_id = file.split(".")[0]
    #     if assay_id not in split_data:
    #         continue
    #     file_path = os.path.join(fsmol_path, file)
    #     with gzip.open(file_path, mode="rt") as f:
    #         ligands = [json.loads(line) for line in f]  # returns a byte string `b'`
    #     ligands = [{"smiles": x["SMILES"],
    #             "pic50_exp": x["LogRegressionProperty"],
    #             "domain": "fsmol"} for x in ligands]
    #     ligand_sets[file] = ligands
    #
    # ret_dict = {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}
    # if not os.path.exists(cache_file):
    #     pickle.dump(ret_dict, open(cache_file, "wb"))
    # return ret_dict


def read_chembl_cell_assay():
    datas = csv.reader(open("/home/fengbin/datas/chembl/chembl_processed_chembl32.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}

    # kd_assay_set = set()
    for line in datas:
        unit = line[7]
        if unit=="%":
            continue
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[9]
        bao_endpoint = line[4]
        bao_format = line[10]
        std_type = line[8]
        # if std_type.lower() != "kd":
        #     continue
        unit = line[7]
        std_rel = line[5]
        if std_rel != "=":
            continue
        is_does = unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                           'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                           "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                           'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
        pic50_exp = -math.log10(float(line[6]))
        affi_prefix = line[5]
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix,
            "is_does": is_does,
            "chembl_assay_type": assay_type,
            "bao_endpoint": bao_endpoint,
            "bao_format": bao_format,
            "unit": unit,
            "domain": "chembl"
        }
        assay_id_dicts[assay_id].append(ligand_info)
    
    # print(list(kd_assay_set))
    # exit()
    assay_id_dicts_new = {}
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.2:
            continue
        if len(ligands) < 20:
            continue
        assay_id_dicts_new[assay_id] = ligands

    return {"ligand_sets": assay_id_dicts_new, "assays": list(assay_id_dicts_new.keys())}


def read_chembl_cell_assay_OOD():
    datas = csv.reader(open("/home/fengbin/datas/chembl/chembl_processed_chembl32_percent.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}
    # kd_assay_set = set()
    pic50s = []
    for line in datas:
        unit = line[7]
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[10]
        std_type = line[8]
        if std_type != "Activity":
            continue
        unit = line[7]
        std_rel = line[5]
        if std_rel != "=":
            continue
        pic50_exp = -math.log10(float(line[6]))
        affi_prefix = line[5]
        pic50s.append(pic50_exp)
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix
        }
        assay_id_dicts[assay_id].append(ligand_info)

    print("ood mean", np.mean(pic50s))
    # print(list(kd_assay_set))
    # exit()
    assay_id_dicts_new = {}
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 20:
            continue
        assay_id_dicts_new[assay_id] = ligands

    assay_ids = list(assay_id_dicts_new.keys())
    random.seed(1111)
    random.shuffle(assay_ids)
    assay_id_dicts_ret = {}
    for k in assay_ids[-500:]:
        assay_id_dicts_ret[k] = assay_id_dicts_new[k]
    return {"ligand_sets": assay_id_dicts_new, "assays": assay_ids[-500:], "test_assay": assay_ids[-500:]}


def read_pQSAR_assay():
    filename = "/home/fengbin/datas/pQSAR/ci9b00375_si_002.txt"
    compound_filename = "/home/fengbin/datas/pQSAR/ci9b00375_si_003.txt"
    # first of all, read all the compounds
    compound_file = open(compound_filename, 'r', encoding='UTF-8', errors='ignore')
    clines = compound_file.readlines()
    compound_file.close()

    import numpy as np
    rng = np.random.RandomState(seed=1111)
    compounds = {}
    previous = ''
    previous_id = ''
    for cline in clines:
        cline = str(cline.strip())
        if 'CHEMBL' not in cline:
            if 'Page' in cline or cline == '' or 'Table' in cline or 'SMILE' in cline:
                continue
            else:
                previous += cline
        else:
            strings = cline.split(',')

            if previous_id not in compounds and previous != '':
                compounds[previous_id] = previous.replace('\u2010', '-')

            previous_id = strings[0]
            previous = strings[1]

    compounds[previous_id] = previous.replace('\u2010', '-')

    assay_ids = []
    ligand_set = {}

    file = open(filename, 'r', encoding='UTF-8', errors='ignore')
    lines = file.readlines()
    file.close()

    for line in lines:
        line = str(line.strip())
        if 'CHEMBL' not in line:
            continue
        strings = line.split(' ')
        compound_id = str(strings[0])
        assay_id = int(strings[1])
        try:
            pic50_exp = float(strings[2])
        except:
            pic50_exp = -float(strings[2][1:])
        train_flag = int(strings[4] == "TRN")

        if assay_id not in assay_ids:
            assay_ids.append(assay_id)

        tmp_example = {
            "affi_prefix": "",
            "smiles": compounds[compound_id],
            "pic50_exp": pic50_exp,
            "train_flag": train_flag,
            "domain": "pqsar"
        }

        if assay_id not in ligand_set:
            ligand_set[assay_id] = []
        ligand_set[assay_id].append(tmp_example)

    return {"ligand_sets": ligand_set,
            "assays": list(ligand_set.keys())}

def read_bdb_cross():
    BDB_all = read_BDB_per_assay()
    save_path = '/home/fengbin/datas/BDB/split_name_train_val_test_bdb.pkl'
    split_name_train_val_test = pickle.load(open(save_path, "rb"))
    repeat_ids = set(
        [x.strip() for x in open("/home/fengbin/meta_delta/scripts/cross_repeat/c2b_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid:BDB_all["ligand_sets"][aid] for aid in test_ids}}

def read_chembl_cross():
    chembl_all = read_chembl_cell_assay()
    save_path = '/home/fengbin/datas/chembl/chembl_split_new.json'
    split_name_train_val_test = json.load(open(save_path, "r"))
    repeat_ids = set(
        [x.strip() for x in open("/home/fengbin/meta_delta/scripts/cross_repeat/b2c_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid:chembl_all["ligand_sets"][aid] for aid in test_ids}}


if __name__ == "__main__":
    datas = read_chembl_cell_assay()["ligand_sets"]
    assay_infos = {}
    units = {}
    for k, v in datas.items():
        ligand_num = len(v)
        bao_endpoint = v[0]["bao_endpoint"]
        bao_format = v[0]["bao_format"]
        assay_type = v[0]["chembl_assay_type"]
        std_type = v[0]["assay_type"]
        unit = v[0]["unit"]
        if unit not in units:
            units[unit] = 0
        units[unit] += 1
        assay_infos[k] = {
            "ligand_num": ligand_num,
            "bao_endpoint": bao_endpoint,
            "bao_format": bao_format,
            "assay_type": assay_type,
            "std_type": std_type,
            "unit": unit
        }
    print(units)
    json.dump(assay_infos, open("assay_infos.json", "w"))
    """BAO_0000218: organism-based format
BAO_0000219: cell-based format
BAO_0000220: subcellular format
BAO_0000221: tissue-based format
BAO_0000223: protein complex format
BAO_0000224: protein format
BAO_0000225: nucleic acid format
BAO_0000249: cell membrane format
BAO_0000357: single protein format"""

    # a = read_chembl_cell_assay_OOD()
    # print(len(a["ligand_sets"]))
    # print(np.mean([len(v) for v in a["ligand_sets"].values()]))
    # unique_ligand = set()
    # for v in a["ligand_sets"].values():
    #     for x in v:
    #         unique_ligand.add(x["smiles"])
    # print(len(unique_ligand))
    # splits = {"train": [], "valid": [], "test": [x for x in a['assays'] if os.path.exists(f"/home/fengbin/datas/expert/davis_feat/{x}.jsonl.gz")]}
    # save_dir = "/home/fengbin/datas/expert/davis_split/split_all_test.json"
    # json.dump(splits, open(save_dir, "w"))

    # stds = []
    # for ligands in a["ligand_sets"].values():
    #     stds.append(np.std([x["pic50_exp"] for x in ligands]))
    # print(np.mean(stds))