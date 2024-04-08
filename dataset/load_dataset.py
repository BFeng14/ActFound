import numpy as np
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
absolute_path = os.path.abspath(__file__)
DATA_PATH = "/".join(absolute_path.split("/")[:-2]+["datas"])

def read_BDB_per_assay(args):
    data_dir = f"{DATA_PATH}/BDB/polymer"
    assays = []
    ligand_sets = {}
    split_cnt = 0
    means = []

    if args.no_fep_lig:
        fep_datas, _ = read_FEP_SET()
        fep_lig_set = set()
        for assay in fep_datas["ligand_sets"].values():
            for lig_info in assay:
                fep_lig_set.add(lig_info["smiles"])

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
                if args.no_fep_lig:
                    if smiles in fep_lig_set:
                        print(smiles, "in fep")
                        continue
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
            means.append(np.mean([x["pic50_exp"] for x in ligands]))
            ligand_sets[assay_name] = ligands

    print(np.mean(means))
    print("split_cnt:", split_cnt)
    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


def read_BDB_IC50():
    data_dir = f"{DATA_PATH}/BDB_baseline"
    ligand_sets = {}
    means = []

    for file_name in tqdm(list(os.listdir(data_dir))):
        assay_name = file_name
        affi_idx = 1
        ligands = []
        affis = []
        file_lines = list(open(os.path.join(data_dir, file_name), "r").readlines())
        for i, line in enumerate(file_lines):
            line = line.strip().split("\t")
            affi_prefix = ""
            pic50_exp = line[8+affi_idx].strip()
            pic50_exp = float(pic50_exp) - 9

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
        if pic50_std == 0.0:
            continue
        if len(ligands) < 20:
            continue
        means.append(np.mean([x["pic50_exp"] for x in ligands]))
        ligand_sets[assay_name] = ligands

    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


def read_gdsc():
    data_file = f"{DATA_PATH}/gdsc/data_dict.pkl"
    ligand_sets = pickle.load(open(data_file, "rb"))
    ligand_sets_new = {}
    for k, v in ligand_sets.items():
        ligand_sets_new[k] = v
        for ligand_info in v:
            ligand_info['pic50_exp'] = -math.log10(math.exp(ligand_info['pic50_exp']))
    print(np.mean([len(x) for x in ligand_sets_new.values()]))
    return {"ligand_sets": ligand_sets_new,
            "assays": list(ligand_sets_new.keys())}


def read_FEP_SET():
    datas = json.load(open(f"{DATA_PATH}/FEP/fep_data_final_norepeat_nocharge.json", "r"))
    ligand_sets = {}
    task2opls4 = {}
    pic50s_all = []
    rmse_all = []
    for k, v in datas.items():
        ligands = []
        opls4_res = []
        errors = []
        for ligand_info in v:
            pic50_exp = -float(ligand_info["exp_dg"]) / 1.379 - 9
            opls4 = -float(ligand_info["pred_dg"]) / 1.379 - 9
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
        rmse = np.sqrt(np.mean(np.square(errors)))
        r2 = np.corrcoef(opls4_res, [x["pic50_exp"] for x in ligands])[0, 1]
        rmse_all.append(r2)
    print("rmse_FEP+(OPLS4)", np.mean(rmse_all))
    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}, task2opls4


def read_kiba():
    ligand_list = []
    ligands_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/kiba/ligands_can.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    Y = pickle.load(open(f"{DATA_PATH}/DeepDTA/data/kiba/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}
    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        ligands = []
        pic50s = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                pic50s.append((affi - 11.72) + -2.24)
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": (affi - 11.72) + -2.24
                    }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        ligand_sets[f"kiba_{assay_idx}"] = ligands



    assay_id_dicts_new = {}
    for assay_id, ligands in ligand_sets.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 50:
            continue
        assay_id_dicts_new[assay_id] = ligands
        stds.append(pic50_std)
    print("stds", np.mean(stds), len(stds))
    return {"ligand_sets": assay_id_dicts_new, "assays": list(assay_id_dicts_new.keys())}


def read_davis():
    ligand_list = []
    protein_list = []
    ligands_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/davis/ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins_dict = json.load(open(f"{DATA_PATH}/DeepDTA/data/davis/proteins.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    for protein_id, seq in proteins_dict.items():
        protein_list.append((protein_id, seq))
    Y = pickle.load(open(f"{DATA_PATH}/DeepDTA/data/davis/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}

    stds = []
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        assay_name, seq = protein_list[assay_idx]
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
        stds.append(len(ligands))
        ligand_sets[f"davis_{assay_idx}"] = ligands

    print("stds", np.mean(stds), len(stds))
    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def read_fsmol_assay(split="train", train_phase=1):
    cache_file = f"{DATA_PATH}/fsmol/{split}_cache.pkl"
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


def read_chembl_assay(args):
    datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl_processed_chembl32.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}

    if args.no_fep_lig:
        fep_datas, _ = read_FEP_SET()
        fep_lig_set = set()
        for assay in fep_datas["ligand_sets"].values():
            for lig_info in assay:
                fep_lig_set.add(lig_info["smiles"])

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
        if args.no_fep_lig:
            if smiles in fep_lig_set:
                continue

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
    datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl_processed_chembl32_percent.csv", "r"),
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
        pic50_exp = math.log10(float(line[6])) - 1.66 + -2.249
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
    ligands_num = []
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.5:
            continue
        if len(ligands) < 20:
            continue
        ligands_num.append(len(ligands))
        assay_id_dicts_new[assay_id] = ligands
    print(np.mean(ligands_num), len(ligands_num))
    assay_ids = list(assay_id_dicts_new.keys())
    return {"ligand_sets": assay_id_dicts_new, "assays": assay_ids}


def read_activity_cliff_assay():
    smiles_as_target = csv.reader(open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/raw_data/all_smiles_target.csv", "r"), delimiter=',')

    assay_dicts = {}
    for line in list(smiles_as_target)[1:]:
        smiles = line[0]
        ki = line[1]
        tid = line[2]

        pic50_exp = -math.log10(float(ki))
        ligand_info = {
            "domain": "activity_cliff",
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": ""
        }

        if tid not in assay_dicts:
            assay_dicts[tid] = {}

        assay_dicts[tid][smiles] = ligand_info

    data_few = json.load(open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Few.json", "r"))
    data_small = json.load(
        open(f"{DATA_PATH}/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Small.json", "r"))
    # data_medium = json.load(
    #     open("../datas/ACNet/ACNet/ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Medium.json", "r"))
    data_all = {**data_few, **data_small}#, **data_medium}

    assay_dicts_processed = {}
    for tid, data in data_all.items():
        ligands = {}
        for pair in data:
            smiles1 = pair["SMILES1"]
            smiles2 = pair["SMILES2"]
            ligands[smiles1] = assay_dicts[tid][smiles1]
            ligands[smiles2] = assay_dicts[tid][smiles2]
        assay_dicts_processed[tid] = {
            "ligands": ligands,
            "pairs": data
        }

    return assay_dicts_processed


def read_pQSAR_assay():
    filename = f"{DATA_PATH}/pQSAR/ci9b00375_si_002.txt"
    compound_filename = f"{DATA_PATH}/pQSAR/ci9b00375_si_003.txt"
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


def read_bdb_cross(args):
    BDB_all = read_BDB_per_assay(args)
    save_path = f'{DATA_PATH}/BDB/bdb_split.json'
    split_name_train_val_test = json.load(open(save_path, "r"))
    repeat_ids = set(
        [x.strip() for x in open(f"{DATA_PATH}/BDB/c2b_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid:BDB_all["ligand_sets"][aid] for aid in test_ids}}


def read_chembl_cross(args):
    chembl_all = read_chembl_assay(args)
    save_path = f'{DATA_PATH}/chembl/chembl_split.json'
    split_name_train_val_test = json.load(open(save_path, "r"))
    repeat_ids = set(
        [x.strip() for x in open(f"{DATA_PATH}/chembl/b2c_repeat", "r").readlines()])
    test_ids = [x for x in split_name_train_val_test['test'] if x not in repeat_ids]
    return {"assays": test_ids, "ligand_sets": {aid:chembl_all["ligand_sets"][aid] for aid in test_ids}}
