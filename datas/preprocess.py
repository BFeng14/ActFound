class Example():
    def __init__(self, smiles, assay_id, pic50_exp, pic50_pred, train_flag):
        self.smiles = smiles
        self.assay_id = assay_id
        self.pic50_exp = pic50_exp
        self.pic50_pred = pic50_pred
        self.split = train_flag


class Experiment():
    def __init__(self, ligand_set, assays, compounds):
        self.ligand_set = ligand_set
        self.assays = assays
        self.compounds = compounds


import numpy as np
import math, os
from tqdm import tqdm
import random
import pickle
def read_BDB(data_dir):
    assays = []
    ligand_sets = {}
    rng = np.random.RandomState(seed=1111)
    save_data = pickle.load(open("./scripts/PubMedBERTFeat.pkl", "rb"))

    assay_descs_list = save_data["assay_descs_list"]
    desc_feats_all = save_data["desc_feats_all"]
    tgt_feats_all = save_data["tgt_feats_all"]
    all_assay_feats = []  #np.concatenate([desc_feats_all, tgt_feats_all], axis=-1)

    assay_path_to_idx = {}
    target_name_dict = {}
    tgt_feats_new = []
    for i, x in enumerate(assay_descs_list):
        assay_path = x[0]
        target_name = x[1][-1]
        if target_name not in target_name_dict:
            target_name_dict[target_name] = len(tgt_feats_new)
            tgt_feats_new.append(tgt_feats_all[i])
        assay_path_to_idx[assay_path] = target_name_dict[target_name]
    tgt_feats_new = np.stack(tgt_feats_new)
    assay_name_to_idx = {}

    for target_name in tqdm(os.listdir(data_dir)):
        for assay_file in os.listdir(os.path.join(data_dir, target_name)):
            assay_name = target_name + "/" + assay_file
            entry_assay = "_".join(assay_file.split("_")[:2])
            assay_path = target_name + "/" + entry_assay
            if assay_path not in assay_path_to_idx:
                continue
            assay_name_to_idx[assay_name] = assay_path_to_idx[assay_path]
            affi_idx = int(assay_file[-5])
            ligands = []
            affis = []
            file_lines = list(open(os.path.join(data_dir, target_name, assay_file), "r").readlines())
            for i, line in enumerate(file_lines):
                line = line.strip().split("\t")
                affi_prefix = ""
                pic50_exp = line[8+affi_idx].strip()
                if pic50_exp.startswith(">") or pic50_exp.startswith("<"):
                    affi_prefix = pic50_exp[1]
                    pic50_exp = pic50_exp[1:]
                pic50_exp = 9 - math.log10(float(pic50_exp))
                smiles = line[1]
                affis.append(pic50_exp)
                ligand_info = {
                    "affi_prefix": affi_prefix,
                    "smiles": smiles,
                    "pic50_exp": pic50_exp
                }
                ligands.append(ligand_info)
            affi_range = np.max(affis) - np.min(affis)
            if affi_range < 1 or affi_range > 5:
                continue
            if len(ligands) >= 300:
                ligands = random.sample(ligands, 300)
            ligand_sets[assay_name] = ligands
            assays.append(assay_name)

    return {"ligand_sets": ligand_sets,
            "assays": assays,
            "assay_name_to_idx": assay_name_to_idx,
            "all_assay_feats": tgt_feats_new}


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
                    pic50_exp = 9 - math.log10(float(pic50_exp))
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
                    "domain": "chembl"
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


import csv
def read_skin_assay_2():
    f = csv.reader(open("/home/fengbin/skin/ChEMBL_sets/assay_CHEMBL4199042.csv", "r"), delimiter=";")

    ligand_info_dict = {}
    for line in f:
        try:
            dg = -math.log10(eval(line[10]))*2
            smiles = line[7]
            ligand_info = {
                "affi_prefix": "",
                "smiles": smiles,
                "pic50_exp": dg
            }
            if smiles not in ligand_info_dict.keys():
                ligand_info_dict[smiles] = ligand_info
            else:
                ligand_info_dict[smiles]["pic50_exp"].append(dg)
        except:
            continue
    for k, v in ligand_info_dict.items():
        v["pic50_exp"] = np.mean(v["pic50_exp"])
    return {"ligand_sets": {"exp_t47d": [x for k,x in ligand_info_dict.items()]},
            "assays": ["exp_t47d"]}


def find_max_affinity_measure(cluster_lines):
    counts_dict = {}
    for line in cluster_lines:
        affi_type = line[5]
        if affi_type not in counts_dict.keys():
            counts_dict[affi_type] = 0
        counts_dict[affi_type] += 1

    counts = sorted([(k,v) for k,v in counts_dict.items()], key=lambda x:x[1], reverse=True)
    return [line for line in cluster_lines if line[5]==counts[0][0]]

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def read_skin_assay_nodesc(file_name):
    f = csv.reader(open(f"/home/fengbin/skin/chembl_nodesc/{file_name}", "r"), delimiter=',', quotechar='"')

    assay_dicts = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        assay_id = line[9]
        if assay_id not in assay_dicts:
            assay_dicts[assay_id] = []
        assay_dicts[assay_id].append(line)
    assay_dicts_new = {}
    # unit_type_set = set()
    density_units = ['uM', 'uM.hr', 'nM', 'mM']
    density_units_2 = ['ug.mL-1', 'ng/ml', 'ug ml-1', 'mg.kg-1']
    measures = ["IC50", "EC50"]

    ligand_info_dict = {}
    for k, v in assay_dicts.items():
        lines_with_val = [x for x in v if len(x[4]) > 0]
        if len(lines_with_val) < 20:
            continue
        lines_with_val = find_max_affinity_measure(lines_with_val)
        unit_type = lines_with_val[0][5]
        measure = lines_with_val[0][6]
        ligands = []
        if unit_type in density_units:
            for line in lines_with_val:
                smile = line[2]
                try:
                    if unit_type == "uM":
                        pic50 = 6-math.log10(float(line[4]))
                    else:
                        pic50 = 9-math.log10(float(line[4]))
                    ligand_info = {
                        "affi_prefix": line[3],
                        "smiles": smile,
                        "pic50_exp": pic50
                    }
                    ligands.append(ligand_info)
                except:
                    continue
        elif unit_type in density_units_2:
            continue
            for line in lines_with_val:
                smile = line[2]
                try:
                    mol = Chem.MolFromSmiles(smile)
                    mw = rdMolDescriptors.CalcExactMolWt(mol)
                    value = float(line[4])/mw
                    pic50 = -math.log10(value)
                    ligand_info = {
                        "affi_prefix": line[3],
                        "smiles": smile,
                        "pic50_exp": pic50
                    }
                    ligands.append(ligand_info)
                except:
                    continue
        else:
            continue
        if len(ligands) < 50:
            continue
        ligand_info_dict[k] = ligands
    return {"ligand_sets": ligand_info_dict,
            "assays": list(ligand_info_dict.keys())}


def read_skin_assay():
    f = csv.reader(open("/home/fengbin/skin/ChEMBL_sets/assay_CHEMBL926551.csv", "r"), delimiter=";")

    ligand_info_dict = {}
    for line in f:
        try:
            dg = 9 + eval(line[10])
            smiles = line[7]
            ligand_info = {
                "affi_prefix": "",
                "smiles": smiles,
                "pic50_exp": dg
            }
            if smiles not in ligand_info_dict.keys():
                ligand_info_dict[smiles] = ligand_info
            else:
                ligand_info_dict[smiles]["pic50_exp"].append(dg)
        except:
            continue
    for k, v in ligand_info_dict.items():
        v["pic50_exp"] = np.mean(v["pic50_exp"])
    return {"ligand_sets": {"exp_t47d": [x for k,x in ligand_info_dict.items()]},
            "assays": ["exp_t47d"]}


def read_pi3ka_t47d():
    f = open("/home/fengbin/DDG/external_test/Pi3ka/exp_t47d.txt", "r").readlines()

    ligand_info_dict = {}
    for i, line in enumerate(f):
        line = line.strip().split("\t")
        try:
            dg = (6 - math.log10(eval(line[2])))
            name = line[0].split(" ")[-1]
            smiles = line[1]
            ligand_info = {
                "affi_prefix": "",
                "smiles": smiles,
                "pic50_exp": [dg],
                "line": i,
                "name": name
            }
            if smiles not in ligand_info_dict.keys():
                ligand_info_dict[smiles] = ligand_info
            else:
                ligand_info_dict[smiles]["pic50_exp"].append(dg)
        except:
            continue
    for k, v in ligand_info_dict.items():
        v["pic50_exp"] = np.mean(v["pic50_exp"])
    return {"ligand_sets": {"exp_t47d": sorted([x for k, x in ligand_info_dict.items()], key=lambda x: x["line"])},
            "assays": ["exp_t47d"]}


def read_pi3ka():
    smiles = open("/home/fengbin/DDG/external_test/Pi3ka/smiles_new.txt", "r").readlines()
    smiles_dict = {}
    for line in smiles:
        line = line.strip().split("\t")
        smile = line[1]
        name = line[0].split(" ")[-1]
        smiles_dict[name] = smile

    ret = []
    f = open("/home/fengbin/DDG/external_test/Pi3ka/exp_all.txt", "r").readlines()

    test_idx = 0
    ligand_sets = {f"Pi3ka_{i}": [] for i in range(test_idx, test_idx+4)}
    for line in f:
        line = line.strip().split("\t")
        for i in range(test_idx, test_idx+4):
            try:
                dg = (9 - math.log10(eval(line[i+1])))
                name = line[0].split(" ")[-1]
                smiles = smiles_dict[name]
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "pic50_exp": dg
                }
                ligand_sets[f"Pi3ka_{i}"].append(ligand_info)
            except:
                continue
    return {"ligand_sets": ligand_sets,
            "assays": [f"Pi3ka_{i}" for i in range(test_idx, test_idx+4)]}


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


def read_covid():
    data_root = "/home/fengbin/datas/covid/processed_data/"
    ligand_sets = {}
    for file_name in os.listdir(data_root):
        if not file_name.endswith("json"):
            continue
        data = json.load(open(os.path.join(data_root, file_name), "r"))
        ligands = []
        for d in data:
            try:
                mol = Chem.MolFromSmiles(d["smiles"], sanitize=True)
                smile = Chem.MolToSmiles(mol)
            except:
                continue
            ligands.append({
                "smiles": smile,
                "pic50_exp": -math.log10(d["AC50"])
            })
        if len(ligands) >= 200:
            for i in range(len(ligands)//200):
                ligand_sets[file_name + f"_{i}"] = ligands[i*200:(i+1)*200]
        else:
            ligand_sets[file_name] = ligands
    print("number of covid assay", len(ligand_sets))
    return {"ligand_sets": ligand_sets,
            "assays": list(ligand_sets.keys())}


import yaml
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
            pic50_exp = -float(ligand_info["exp_dg"]) / 1.38
            opls4 = -float(ligand_info["pred_dg"]) / 1.38
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

    # def parse_finetune_file_v2(path):
    #     f = yaml.safe_load(open(path, "r"))
    #     ret = []
    #     for name, v in f.items():
    #         dg = v["measurement"]["value"]
    #         unit = v["measurement"]["unit"]
    #         type = v["measurement"]["type"]
    #         smiles = v["smiles"]
    #         if unit.lower() == "nm":
    #             dg = 9 - math.log10(dg)
    #         elif unit.lower() == "um":
    #             dg = 6 - math.log10(dg)
    #         elif unit.lower() == "kilojoules / mole":
    #             dg = -dg / 1.358
    #         elif type.lower() == "pic50":
    #             dg = dg
    #         else:
    #             exit()
    #         ret.append((name, dg, smiles))
    #     return ret
    #
    # root = "/home/fengbin/datas/FEP/data"
    # ligand_sets = {}
    # assays = []
    # pic50s = []
    # merck_list = ["2020-07-06_pfkfb3", "2020-07-03_hif2a", "2019-12-13_cmet", "2020-08-11_syk", "2020-07-01_cdk8", "2020-08-12_tnks2", "2020-06-19_eg5", "2020-07-30_shp2"]
    # schordinger_list = ["2020-02-04_bace", "2019-12-13_mcl1", "2019-12-13_cdk2", "2019-09-23_jnk1", "2019-12-09_p38", "2019-12-12_ptp1b", "2020-02-07_tyk2", "2019-09-23_thrombin"]
    # for t_name in merck_list + schordinger_list:
    #     ligands = []
    #     type_file = f"{root}/{t_name}/00_data/ligands.yml"
    #     data_info = parse_finetune_file_v2(type_file)
    #     train_cnt = 0
    #     for ligand in data_info:
    #         pic50_exp = ligand[1]
    #         smiles = ligand[2]
    #         pic50s.append(pic50_exp)
    #         ligand_info = {
    #             "affi_prefix": "",
    #             "smiles": smiles,
    #             "pic50_exp": pic50_exp,
    #             "domain": "fep"
    #         }
    #         train_cnt += 1
    #         ligands.append(ligand_info)
    #     # if len(ligands) <= 20:
    #     #     continue
    #     ligand_sets[t_name] = ligands
    #     assays.append(t_name)
    #
    # print("mean", np.mean(pic50s))
    # return {"ligand_sets": ligand_sets, "assays": assays}

from collections import OrderedDict
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
                pic50s.append(9-math.log10(affi))
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": 9-math.log10(affi)
                    }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        stds.append(np.std(pic50s))
        ligand_sets[f"davis_{assay_idx}"] = ligands

    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def read_BDB_interaction_merck(avg, std):
    def parse_finetune_file_v2(path):
        f = yaml.safe_load(open(path, "r"))
        ret = []
        for name, v in f.items():
            dg = v["measurement"]["value"]
            unit = v["measurement"]["unit"]
            type = v["measurement"]["type"]
            smiles = v["smiles"]
            if unit.lower() == "nm":
                dg = 9 - math.log10(dg)
            elif unit.lower() == "um":
                dg = 6 - math.log10(dg)
            elif unit.lower() == "kilojoules / mole":
                dg = -dg / 1.358
            elif type.lower() == "pic50":
                dg = dg
            else:
                exit()
            ret.append((name, dg, smiles))
        return ret

    root = "/home/fengbin/protein-ligand-benchmark/extracted_vol"
    ligand_sets = {}
    assays = []
    merck_list = ["2020-07-06_pfkfb3", "2020-07-03_hif2a", "2019-12-13_cmet", "2020-08-11_syk", "2020-07-01_cdk8", "2020-08-12_tnks2", "2020-06-19_eg5", "2020-07-30_shp2"]
    schordinger_list = ["2020-02-04_bace", "2019-12-13_mcl1", "2019-12-13_cdk2", "2019-09-23_jnk1", "2019-12-09_p38", "2019-12-12_ptp1b", "2020-02-07_tyk2", "2019-09-23_thrombin"]
    for t_name in merck_list:
        ligands = []
        ligand_root_path = f"/home/fengbin/protein-ligand-benchmark/data/{t_name}/vina_fp"
        type_file = f"{root}/{t_name}/ligands.yml"
        data_info = parse_finetune_file_v2(type_file)
        random.seed(2013)
        random.shuffle(data_info)
        train_cnt = 0
        for ligand in data_info:
            ligand_name = ligand[0]
            pic50_exp = ligand[1]
            smiles = ligand[2]
            ligand_fp_file = os.path.join(ligand_root_path, "opt_" + ligand_name.split(".")[0], "Input_min.csv")
            if os.path.exists(ligand_fp_file):  # and os.path.exists(ligand_splif_path):
                ligand_fp = list(open(ligand_fp_file, "r").readlines())[1]
                ligand_fp = ligand_fp.strip().split(",")[1:]
                ligand_fp = np.array([float(x) for x in ligand_fp])
                ligand_info = {
                    "smiles": smiles,
                    "pic50_exp": pic50_exp,
                    "feature": (ligand_fp-avg)/(std+1e-5),
                    "split": train_cnt < int(0.75*len(data_info))
                }
                train_cnt += 1
                ligands.append(ligand_info)
        ligand_sets[t_name] = ligands
        assays.append(t_name)

    return {"ligand_sets": ligand_sets, "assays": assays}


def read_BDB_interactionFP():
    def parse_data_info():
        info_file = "/home/fengbin/DDG/BindingDB_data/jimenez/jl_comparison_set_new.csv"
        data_info = {}
        info_file = list(open(info_file, "r").readlines())[1:]
        for line in info_file:
            line = line.strip().split(",")
            lig_file = line[3].split("/")[-1]
            smile = line[5]
            dg = 9 - math.log10(eval(line[4]))
            target = line[2]
            if target not in data_info.keys():
                data_info[target] = []
            data_info[target].append((lig_file, dg, smile))
        return data_info

    assays = []
    ligand_sets = {}
    ligand_fps = []
    data_info = parse_data_info()
    for target in tqdm(data_info.keys()):
        target_name = target
        ligands = []
        for ligand in data_info[target]:
            ligand_name = ligand[0]
            pic50_exp = ligand[1]
            smiles = ligand[2]
            ligand_fp_file = os.path.join("/home/fengbin/surflex_data/minimize_results/", target,
                                          "opt_" + ligand_name.split(".")[0], "Input_min.csv")

            if os.path.exists(ligand_fp_file):
                ligand_fp = list(open(ligand_fp_file, "r").readlines())[1]
                ligand_fp = ligand_fp.strip().split(",")[1:]
                ligand_fp = np.array([float(x) for x in ligand_fp])
                ligand_info = {
                    "smiles": smiles,
                    "pic50_exp": pic50_exp,
                    "feature": ligand_fp
                }
                ligand_fps.append(ligand_fp)
                ligands.append(ligand_info)
        if len(ligands) < 10:
            continue
        sups_idx = set(random.sample(range(len(ligands)), int(0.7 * len(ligands))))  #
        for i, ligand_info in enumerate(ligands):
            split = i in sups_idx
            ligand_info["split"] = split
        ligand_sets[target_name] = ligands
        assays.append(target_name)

    ligand_fps = np.array(ligand_fps)
    avg = np.mean(ligand_fps, axis=0)
    std = np.std(ligand_fps, axis=0)
    for target_name, ligands in ligand_sets.items():
        for ligand_info in ligands:
            ligand_info["feature"] = (ligand_info["feature"]-avg)/(std+1e-5)

    return {"ligand_sets": ligand_sets, "assays": assays}, avg, std

def read_skin_assay_ids():
    assay_id_set = set()
    for f in os.listdir("/cpfs01/user/Fengbin2014/meta/datas/chembl/smi_chembl32/test_assay"):
        # if f not in ["ch32.atph.csv", "ch32.derm.csv"]:
        #     continue
        f = os.path.join("/cpfs01/user/Fengbin2014/meta/datas/chembl/smi_chembl32/test_assay", f)
        for i, line in enumerate(open(f, "r").readlines()):
            if i == 0:
                continue
            assay_id = line.strip().split(',')[2]
            assay_id_set.add(assay_id)
    return assay_id_set

import json, pickle
import gzip
def read_fsmol_assay(split = "train", train_phase=1):
    cache_file = f"/home/fengbin/datas/fsmol/{split}_cache.pkl"
    if os.path.exists(cache_file):
        return pickle.load(open(cache_file, "rb"))

    if train_phase == 0 and split == "train":
        return {"ligand_sets": {}, "assays": []}
    fsmol_path = f"./fsmol_data/{split}"
    ligand_sets = {}
    if split == "test":
        test_file = open("./fsmol_data/regression_test_adkfift.csv", "r").readlines()
        split_data = [x.split(",")[0] for x in test_file][1:]
    else:
        split_path = json.load(open("./fsmol_data/fsmol-0.1.json", "r"))
        split_data = split_path[split]
    for file in tqdm(os.listdir(fsmol_path)):
        assay_id = file.split(".")[0]
        if assay_id not in split_data:
            continue
        file_path = os.path.join(fsmol_path, file)
        with gzip.open(file_path, mode="rt") as f:
            ligands = [json.loads(line) for line in f]  # returns a byte string `b'`
        ligands = [{"SMILES": x["SMILES"], 
                "LogRegressionProperty": x["LogRegressionProperty"],
                "Property": x["Property"]} for x in ligands]
        ligand_sets[file] = ligands

    ret_dict = {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}
    # if not os.path.exists(cache_file):
    #     pickle.dump(ret_dict, open(cache_file, "wb"))
    return ret_dict

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
            "unit": unit
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


def read_chembl_covid_assay():
    datas = list(csv.reader(open("/home/fengbin/datas/covid/covid_activity.csv", "r"), delimiter=';'))
    datas_dict = []
    for d in datas[1:]:
        datas_dict.append({k:v for k, v in zip(datas[0], d)})

    assay_id_dicts = {}
    # kd_assay_set = set()
    for line in datas_dict:
        unit = line["Standard Units"]
        std_type = line["Standard Type"]
        assay_id = "{}_{}_{}".format(line["Assay ChEMBL ID"], unit, std_type).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line["Smiles"]

        std_rel = line["Standard Relation"]
        if std_rel != "'='":
            continue
        is_does = unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                           'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                           "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                           'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
        if unit == "%":
            try:
                pic50_exp = math.log10(float(line["Standard Value"]))
            except:
                continue
        elif unit in ['uM', 'nM']:
            try:
                pic50_exp = -math.log10(float(line["Standard Value"]))
            except:
                continue
        else:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            smiles = Chem.MolToSmiles(mol)
        except:
            continue
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": std_rel,
            "is_does": is_does
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
            "train_flag": train_flag
        }

        if assay_id not in ligand_set:
            ligand_set[assay_id] = []
        ligand_set[assay_id].append(tmp_example)

    return {"ligand_sets": ligand_set,
            "assays": list(ligand_set.keys())}


def read_4276_txt(filename, compound_filename):
    # first of all, read all the compounds
    compound_file = open(compound_filename, 'r', encoding='UTF-8', errors='ignore')
    clines = compound_file.readlines()
    compound_file.close()

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
    real_train = {}
    real_test = {}

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
        try:
            pic50_pred = float(strings[3])
        except:
            pic50_pred = -float(strings[3][1:])
        train_flag = strings[4]

        if assay_id not in assay_ids:
            assay_ids.append(assay_id)

        tmp_example = Example(compound_id, assay_id, pic50_exp, pic50_pred)

        if assay_id not in ligand_set:
            ligand_set[assay_id] = []
            ligand_set[assay_id].append(tmp_example)
        else:
            ligand_set[assay_id].append(tmp_example)

    experiment = Experiment(ligand_set, assay_ids, compounds)

    return experiment

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