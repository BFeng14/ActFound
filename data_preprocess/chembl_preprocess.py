import csv
import os
absolute_path = os.path.abspath(__file__)
DATA_PATH = "/" + "/".join(absolute_path.split("/")[:-2]+["datas"])

datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl32_0307/cnt_actv.csv", "r"), delimiter=',')

selected_units_c = ['nM', 'uM', 'mm', 'umol.kg-1', 'mM', 'uM kg-1', "10'-2umol", # 47776
                  'M', 'nM kg-1', 'um', 'umol/Kg', 'nm', "10'-6M", "10'-7M", '10^-2mm', 'nmol', ] # 47966
selected_units_m = ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                    'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                    "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                    'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
selected_units_p = ["%"]

smile_dict = {}
smile_csv = csv.reader(open(f"{DATA_PATH}/chembl/smi_chembl32/molsmi.csv", "r"), delimiter=',')
for line in smile_csv:
    molregno = line[0]
    canonical_smiles = line[1]
    smile_dict[molregno] = canonical_smiles

assay_id_dicts = {}
for line in datas:
    std_units_raw = line[7]
    if std_units_raw not in selected_units_c + selected_units_m:
        continue
    assay_id = "{}_{}_{}".format(line[1], line[7], line[8])
    molregno = line[3]
    smile = smile_dict.get(molregno, None)
    if smile is not None:
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        line.append(smile)
        assay_id_dicts[assay_id].append(line)

import numpy as np
def remove_duplicate_ligands(cluster_lines):
    monomer_id_dict = {}
    for line in cluster_lines:
        monomer_id = line[3]
        if monomer_id not in monomer_id_dict.keys():
            monomer_id_dict[monomer_id] = []
        monomer_id_dict[monomer_id].append(line)

    new_cluster_lines = []
    for monomer_id, lines in monomer_id_dict.items():
        if len(lines) == 1:
            line_avg = lines[0]
        else:
            affinitys_all = []
            for line in lines:
                affinity = line[6].strip()
                affinity = float(affinity)
                affinitys_all.append(affinity)
            affinitys_all = sorted(affinitys_all)
            if affinitys_all[0] <= 0:
                continue
            # remove any data have multiaffi, and the range is more than one order
            if affinitys_all[-2] > 10*affinitys_all[0]:
                continue
            else:
                # compute the average value if have multiple value
                avg_affinity = np.mean(affinitys_all)
                line_avg = lines[0]
                line_avg[6] = str(avg_affinity)
        # remove any data that <= 0
        if float(line_avg[6]) <= 0:
            continue
        new_cluster_lines.append(line_avg)
    return new_cluster_lines


assay_id_dicts_new = {}
for assay_id, lines in assay_id_dicts.items():
    example_line = lines[0]
    std_units_raw = example_line[7]
    if len(lines) < 20:
        continue
    if std_units_raw in selected_units_p:
        values_list = [float(line[6]) for line in lines]
        if min(values_list) < 0:
            continue
    lines_nodup = remove_duplicate_ligands(lines)
    if len(lines_nodup) < 20:
        continue
    for line in lines_nodup:
        line[-2] = len(lines_nodup)
    assay_id_dicts_new[assay_id] = lines_nodup

assay_cnt = [len(x) for x in assay_id_dicts_new.values()]
print(len([x for x in assay_cnt if x >= 10000]))
print(len([x for x in assay_cnt if x >= 30000]))
print(len([x for x in assay_cnt if x >= 60000]))

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import math
from tqdm import tqdm
from multiprocessing import Pool

def construct_graphs(lines):
    if len(lines) >= 10000:
        return None, None
    fps = []
    affis = []
    for line in lines:
        smiles = line[-1]
        arr = fp_dict.get(smiles, None)
        if arr is None:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useChirality=True)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
        else:
            fp_string = "".join([str(x) for x in arr.tolist()])
            fp = DataStructs.CreateFromBitString(fp_string)
        fps.append(fp)
        affis.append(-math.log10(float(line[6])))

    degree = [0 for _ in range(len(fps))]
    pairs = []
    for i in range(len(fps) - 1):
        sim = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[(i + 1):]))
        sim_thres = 0.3
        sim_num = np.sum(sim >= sim_thres)
        if sim_num == 0:
            idx = []
        else:
            idx = np.where(sim >= sim_thres)[0]

        idx_sim = [(i + 1 + x, sim[x]) for x in idx]
        sim_pair = [(i, x[0], x[1]) for x in idx_sim]

        for l_pair in sim_pair:
            a, b, sim = l_pair
            a_prefix = lines[a][5]
            b_prefix = lines[b][5]
            if a_prefix == b_prefix == "<":
                continue
            if a_prefix == b_prefix == ">":
                continue
            if affis[a] - affis[b] == 0:
                if sim <= 0.7:
                    continue
            degree[a] += 1
            degree[b] += 1
            pairs.append(l_pair)
    return degree, len(pairs)

chunks = list(assay_id_dicts_new.keys())
fp_dict = {}
assay_id_dicts_final = {}
len_pair_all = []
with Pool(16) as p:
    res_all = p.map(construct_graphs, tqdm([assay_id_dicts_new[x] for x in chunks]))

    for res, assay_id in tqdm(zip(res_all, chunks)):
        degree = res[0]
        if degree is None:
            continue
        len_pair = res[1]
        len_pair_all.append(len_pair)
        lines = assay_id_dicts_new[assay_id]
        valid_cnt = [x for x in degree if x>= math.sqrt(math.sqrt(len(lines))) * 2]
        lines_new = [lines[i] for i in range(len(lines)) if degree[i]>0]
        if len(valid_cnt) < 20:
            continue
        for line in lines_new:
            line[-2] = len(lines_new)
            line.append(len_pair)
        assay_id_dicts_final[assay_id] = lines_new

print(max(len_pair_all), min(len_pair_all))
# import pickle
# pickle.dump(fp_dict, open("/home/fengbin/CHEMBL/chembl_fp_cache.pkl", "wb"))
with open(f"{DATA_PATH}/chembl/chembl_processed_chembl32.csv", "w") as f:
    writer = csv.writer(f)
    for assay_id, lines in assay_id_dicts_final.items():
        writer.writerows(lines)

assay_cnt = [len(x) for x in assay_id_dicts_final.values()]
for thres in [100, 300, 1000, 3000, 10000]:
    tmp = [x for x in assay_cnt if x <=thres]
    num_assay = len(tmp)
    num_data = sum(tmp)
    print("thres: {}, number of assay: {}, number of data: {}, average: {}".format(thres, num_assay, num_data, num_data/num_assay))