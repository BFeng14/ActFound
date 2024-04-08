import copy
import os
import numpy as np
import tqdm
import pickle
import torch
import dataset.load_dataset as preprocess
import random
from multiprocessing import Pool
from dataset.data_base import preprocess_assay, BaseMetaDataset
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def preprocess_assay_cliff(data):
    x_tmp = []
    smiles_list = []
    affis = []

    lines, split = data
    if lines is None:
        return None

    if len(lines) > 10000:
        return None

    for line in lines:
        smiles = line["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # mol = Chem.RemoveHs(mol)
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)

        pic50_exp = line["pic50_exp"]
        affis.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(affis).astype(np.float32)
    return x_tmp, affis, smiles_list, split


class ActCliffMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        self.shot_num = 16
        super(ActCliffMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        experiment_actcliff = preprocess.read_activity_cliff_assay()

        def add_neighbor(neib, k, v):
            if k not in neib:
                neib[k] = set()
            neib[k].add(v)

        assays_new = {}
        for assay_id, assay_info in experiment_actcliff.items():
            ligands_dict = assay_info["ligands"]
            pairs = assay_info["pairs"]
            mol_have_cliff = set()
            neighbor_dict = {}
            neighbor_cliff_dict = {}
            for pair in pairs:
                smiles1 = pair["SMILES1"]
                smiles2 = pair["SMILES2"]
                add_neighbor(neighbor_dict, smiles1, smiles2)
                add_neighbor(neighbor_dict, smiles2, smiles1)
                value = pair["Value"]
                if value == '1':
                    add_neighbor(neighbor_cliff_dict, smiles1, smiles2)
                    add_neighbor(neighbor_cliff_dict, smiles2, smiles1)
                    mol_have_cliff.add(smiles1)
                    mol_have_cliff.add(smiles2)

            mol_have_cliff = sorted(list(mol_have_cliff))
            for mol, neighbors in neighbor_dict.items():
                neighbor_dict[mol] = sorted(list(neighbors))

            for i, mol in enumerate(mol_have_cliff):
                # activity cliff
                tgt_mols = list(neighbor_cliff_dict[mol])
                neighbors = copy.deepcopy(tgt_mols)
                # for t_mol in tgt_mols:
                #     neighbors += neighbor_dict[t_mol]

                non_neighbors = [s for s in ligands_dict.keys() if s not in neighbors + [mol]]
                # random.seed(i)
                # random.shuffle(non_neighbors)
                sup_mols = [mol] + non_neighbors

                split = [1] * len(sup_mols) + [0] * len(tgt_mols)

                mol_infos = [ligands_dict[x] for x in sup_mols+tgt_mols]
                assays_new[f"{assay_id}_{i}"] = (mol_infos, split)


        self.assay_ids = list(assays_new.keys())
        self.n_assays = len(assays_new)

        self.indices = []
        self.Xs = []
        self.smiles_all = []
        self.split_all = []
        self.ys = []
        self.assaes = []
        self.sims = []
        self.degrees = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.test_scaffold_split = {}

        assay_list = self.assay_ids

        data_cnt = 0
        with Pool(8) as p:
            res_all = p.map(preprocess_assay_cliff, tqdm.tqdm([assays_new[x] for x in self.assay_ids]))
            for res, assay_id in zip(res_all, assay_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list, split = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
                self.split_all.append(split)
                self.assaes.append(assay_id)
                self.test_indices.append(data_cnt)
                data_cnt += 1

        test_cnt = len(self.test_indices)

        self.data_length = {}
        self.data_length['train'] = 0
        self.data_length['val'] = 0
        self.data_length['train_weight'] = 0
        self.data_length['test'] = test_cnt

        print(np.max([len(x) for x in self.Xs]), np.mean([len(x) for x in self.Xs]))

    def get_split(self, X_in, y_in, scaffold_split=None, **kwargs):
        X, y = X_in, y_in
        split = np.array(scaffold_split)
        return [X, y, split]
