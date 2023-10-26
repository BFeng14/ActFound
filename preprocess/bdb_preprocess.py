import os.path
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

f = open("/home/fengbin/datas/BDB/BindingDB_All.tsv", "r").readlines()
target_link_set = set()

from tqdm import tqdm

def get_dist(fps):

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    return dists


def get_entry_assay_dict():
    f = "/home/fengbin/datas/BDB/ENTRY_ASSAY_export_2022-12-26_120436.csv"
    ret_dict = {}
    for i, line in enumerate(open(f, "r").readlines()):
        if i == 0:
            continue
        REACTANT_SET_ID, ENTRYID, ASSAYID = tuple(line.strip().split(","))
        if len(ENTRYID) > 0 and len(ASSAYID) > 0:
            ret_dict[REACTANT_SET_ID] = f"{ENTRYID}_{ASSAYID}"
    return ret_dict


def find_max_affinity_measure(cluster_lines):
    counts = [0, 0, 0] # Ki, IC50, Kd
    for line in cluster_lines:
        affinitys = line.split("\t")[8:11]
        affi_type = [len(x[:1]) for x in affinitys]
        counts = [x+y for x, y in zip(counts, affi_type)]

    max_idx = 2
    max_num = counts[2]
    if counts[0] > max_num:
        max_idx = 0
        max_num = counts[0]
    if counts[1] > max_num*2:
        max_idx = 1
        max_num = counts[1]

    cluster_lines_new = []
    for line in cluster_lines:
        affinitys = line.split("\t")[8:11]
        affi_type = [len(x.strip()[:1]) for x in affinitys]
        if affi_type[max_idx] > 0:
            cluster_lines_new.append(line)

    return cluster_lines_new, max_idx


# from statistics import geometric_mean
import numpy as np
def remove_duplicate_ligands(cluster_lines, max_idx):
    smiles_dict = {}
    for line in cluster_lines:
        smiles = line.split("\t")[1]
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            canon_smiles = Chem.MolToSmiles(mol, True)
            # fp = Chem.RDKFingerprint(mol)
            # arr = np.zeros((0,), dtype=np.int8)
            # DataStructs.ConvertToNumpyArray(fp, arr)
            # fp_cache[canon_smiles] = arr
        except:
            continue
        if canon_smiles not in smiles_dict.keys():
            smiles_dict[canon_smiles] = []
        smiles_dict[canon_smiles].append(line)

    new_cluster_lines = []
    for canon_smiles, lines in smiles_dict.items():
        if len(lines) == 1:
            new_cluster_lines.append(lines[0])
        else:
            affinitys_all = []
            for line in lines:
                affinity = line.split("\t")[8:11][max_idx].strip()
                if affinity.startswith(">") or affinity.startswith("<"):
                    affinity = affinity[1:]
                affinity = float(affinity)
                affinitys_all.append(affinity)
            affinitys_all = sorted(affinitys_all)
            if affinitys_all[-1]/affinitys_all[0] > 10:
                continue
            else:
                avg_affinity = 10**np.log10(affinitys_all).mean()
                line_avg = lines[0].split("\t")
                line_avg[8+max_idx] = str(avg_affinity)
                line_avg[1] = canon_smiles
                new_cluster_lines.append("\t".join(line_avg))
    return new_cluster_lines

per_target_entry_assay_dict = {}
entry_assay_dict = get_entry_assay_dict()

for i, line in tqdm(enumerate(f)):
    if i == 0:
        continue
    line = line.strip()
    ligand_smile = line.split("\t")[1]
    target_line = line.strip().split("\t")[24]
    if "polymerid=" in target_line:
        tmp = target_line.find("polymerid=")
        tgt_str = target_line[tmp:].split("&")[0]
    elif "complexid=" in target_line:
        tmp = target_line.find("complexid=")
        tgt_str = target_line[tmp:].split("&")[0]
    else:
        continue

    entry_assay = entry_assay_dict.get(line.split("\t")[0], "")
    insti = line.split("\t")[22].strip()
    affinitys = line.split("\t")[8:11]
    affi_type = str([len(x[:1]) for x in affinitys])

    # 去除没有IC50, Ki, Kd的数据
    if affi_type == "[0, 0, 0]":
        continue

    # 去除没有institution和assay description的数据
    if insti == "TBA":
        continue
    if len(entry_assay) == 0 or len(insti) == 0:
        continue

    if tgt_str not in per_target_entry_assay_dict.keys():
        per_target_entry_assay_dict[tgt_str] = {}
    if entry_assay not in per_target_entry_assay_dict[tgt_str].keys():
        per_target_entry_assay_dict[tgt_str][entry_assay] = []
    per_target_entry_assay_dict[tgt_str][entry_assay].append(line)

import pickle
total_cluster_num = 0
total_data_num = 0
# ligand_fp_dict = pickle.load(open("../data_curate/ligand_fp_dict.pkl", "rb"))
fp_cache = {}

for tgt_str, assays in tqdm(per_target_entry_assay_dict.items()):
    for entry_assay, lines in assays.items():
        id = tgt_str.split("=")[-1]

        fasta_seq = lines[0].split("\t")[37]
        if len(fasta_seq) <= 40:
            print("too small prptein:", tgt_str)
            continue

        example_line = lines[0]
        target_name = example_line.split("\t")[6]
        target_name = target_name.replace("/", "|").replace("(", "|").replace(")", "|").replace("{", "|").replace("}", "|").strip()
        target_name = "-".join(target_name.split(" "))

        if "polymerid" in tgt_str:
            save_dir = f"/home/fengbin/datas/BDB/polymer/{target_name}"
        elif "complexid" in tgt_str:
            save_dir = f"/home/fengbin/datas/BDB/complex/{target_name}"
        else:
            print("bugggggg", tgt_str)
            continue

        if len(lines) < 25:
            continue
        lines, max_idx = find_max_affinity_measure(lines)
        lines = remove_duplicate_ligands(lines, max_idx)
        if len(lines) < 25:
            continue

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/{entry_assay}_{max_idx}.tsv", "w") as wf:
            for line in lines:
                wf.write(line + "\n")

        total_cluster_num += 1
        total_data_num += len(lines)

print("total cluster num", total_cluster_num)
print("total data num", total_data_num)