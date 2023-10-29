import csv
import numpy as np
import math
import os
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool
import pickle

def read_chembl_cell_assay():
    datas = csv.reader(open("../datas/chembl/chembl_processed_chembl32.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}
    # kd_assay_set = set()
    for line in datas:
        unit = line[7]
        if unit == "%":
            continue
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[10]
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
        # try:
        #     mol = Chem.MolFromSmiles(smiles, sanitize=True)
        #     smiles_new = Chem.MolToSmiles(mol)
        #     assert smiles_new == smiles
        # except:
        #     continue

        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix,
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


def read_BDB_per_assay():
    data_dir = "../datas/BDB/polymer"
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
                pic50_exp = line[8 + affi_idx].strip()
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
                    "pic50_exp": pic50_exp
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


from collections import OrderedDict
import json
def read_davis():
    ligand_list = []
    ligands_dict = json.load(open("../datas/DeepDTA/data/davis/ligands_can.txt"), object_pairs_hook=OrderedDict)
    for ligand_id, smiles in ligands_dict.items():
        ligand_list.append((ligand_id, smiles))
    Y = pickle.load(open("../datas/DeepDTA/data/davis/Y", "rb"), encoding='bytes').transpose()
    ligand_sets = {}
    for assay_idx in range(Y.shape[0]):
        affis = Y[assay_idx]
        ligands = []
        for i, affi in enumerate(affis):
            if not np.isnan(affi) and affi < 10000:
                ligand_id, smiles = ligand_list[i]
                ligand_info = {
                    "affi_prefix": "",
                    "smiles": smiles,
                    "ligand_id": ligand_id,
                    "pic50_exp": 9-math.log10(affi)
                    }
                ligands.append(ligand_info)
        if len(ligands) < 20:
            continue
        ligand_sets[f"davis_{assay_idx}"] = ligands

    return {"ligand_sets": ligand_sets, "assays": list(ligand_sets.keys())}


def construct_pairs(lines):
    smiles_dict = {}

    if len(lines) > 10000:
        return None
    for line in lines:
        smiles = line["smiles"]

        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            smiles = Chem.MolToSmiles(mol)
        except:
            continue
        pic50_exp = line["pic50_exp"]
        smiles_dict[smiles] = pic50_exp

    return smiles_dict


bdb_set = read_BDB_per_assay()
chembl_set = read_chembl_cell_assay()

davis_set = read_davis()

chembl_canismiles = {}
with Pool(16) as p:
    res_all = p.map(construct_pairs, tqdm([chembl_set["ligand_sets"][x] for x in chembl_set["assays"]]))
    for res, assay_id in zip(res_all, chembl_set["assays"]):
        chembl_canismiles[assay_id] = res

davis_canismiles = {}
with Pool(16) as p:
    res_all = p.map(construct_pairs, tqdm([davis_set["ligand_sets"][x] for x in davis_set["assays"]]))
    for res, assay_id in zip(res_all, davis_set["assays"]):
        davis_canismiles[assay_id] = res

bdb_canismiles = {}
with Pool(16) as p:
    res_all = p.map(construct_pairs, tqdm([bdb_set["ligand_sets"][x] for x in bdb_set["assays"]]))
    for res, assay_id in zip(res_all, bdb_set["assays"]):
        bdb_canismiles[assay_id] = res


print("repeating assay on bindingdb")
repeat_set_on_bdb = set()
for davisid in davis_canismiles.keys():
    davis_ligands = davis_canismiles[davisid]
    davis_ligands_set = set(list(davis_ligands.keys()))
    for bid, test_ligands in bdb_canismiles.items():
        test_ligands = test_ligands
        test_ligands_set = set(list(test_ligands.keys()))
        repeat = davis_ligands_set.intersection(test_ligands_set)
        if len(repeat) >= 4:
            bdb_ys = [test_ligands[x] for x in repeat]
            davis_ys = [davis_ligands[x] for x in repeat]
            corr = np.corrcoef(bdb_ys, davis_ys)[0, 1]
            if corr >= 0.99:
                repeat_set_on_bdb.add(bid)
                print(davisid, bid, len(repeat), corr)


print("repeating assay on chembl")
repeat_set_on_chembl = set()
for davisid in davis_canismiles.keys():
    davis_ligands = davis_canismiles[davisid]
    davis_ligands_set = set(list(davis_ligands.keys()))
    for cid, test_ligands in chembl_canismiles.items():
        test_ligands = test_ligands
        test_ligands_set = set(list(test_ligands.keys()))
        repeat = davis_ligands_set.intersection(test_ligands_set)
        if len(repeat) >= 4:
            chembl_ys = [test_ligands[x] for x in repeat]
            davis_ys = [davis_ligands[x] for x in repeat]
            corr = np.corrcoef(chembl_ys, davis_ys)[0, 1]
            if corr >= 0.99:
                repeat_set_on_chembl.add(cid)
                print(davisid, cid, len(repeat), corr)

json.dump(list(repeat_set_on_chembl), open("davis_repeat_set_on_chembl.txt", "w"))
json.dump(list(repeat_set_on_bdb), open("davis_repeat_set_on_bdb.txt", "w"))