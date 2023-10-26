import json
import math
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
import dataset.load_dataset as preprocess
import copy
import random
from multiprocessing import Pool
from dataset.data_base import preprocess_assay, BaseMetaDataset
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


def preprocess_assay_pqsar(lines):
    x_tmp = []
    smiles_list = []
    affis = []
    split = []

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
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useChirality=True)
        # arr = np.zeros((0,), dtype=np.int8)
        # DataStructs.ConvertToNumpyArray(fp, arr)
        pic50_exp = line["pic50_exp"]
        affis.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)
        split.append(line.get("train_flag", None))

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(affis).astype(np.float32)
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb', 'none']:
        return None
    return x_tmp, affis, smiles_list, split


class pQSARMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(pQSARMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        assert self.args.datasource == "pqsar"
        experiment_train = preprocess.read_pQSAR_assay()

        self.assay_ids = experiment_train["assays"]

        ligand_set = experiment_train["ligand_sets"]
        self.n_assays = len(ligand_set)

        self.indices = []
        save_dir = '{0}/{1}'.format(self.args.logdir, self.exp_string)
        
        if not os.path.exists(save_dir):
            os.system(f"mkdir -p {save_dir}")

        self.split_name_train_val_test = pickle.load(open(f"{DATA_PATH}/pQSAR/drug_split_id_group1.pickle", "rb"))
        self.split_name_train_val_test["valid"] = self.split_name_train_val_test["val"]

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

        ma_pair = 0
        assay_list = []

        assay_list += self.split_name_train_val_test['test']
        # valid set
        assay_list += self.split_name_train_val_test['valid']
        # train set
        if self.args.train == 1 or self.args.knn_maml:
            assay_list += self.split_name_train_val_test['train']

        data_cnt = 0
        with Pool(16) as p:
            res_all = p.map(preprocess_assay_pqsar, tqdm.tqdm([ligand_set.get(x, None) for x in assay_list]))

            for res, assay_id in zip(res_all, assay_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list, split = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
                self.split_all.append(split)
                self.assaes.append(assay_id)
                if assay_id in self.split_name_train_val_test['train']:
                    self.train_indices.append(data_cnt)
                    data_cnt += 1
                elif assay_id in self.split_name_train_val_test['valid']:
                    self.val_indices.append(data_cnt)
                    data_cnt += 1
                elif assay_id in self.split_name_train_val_test['test']:
                    self.test_indices.append(data_cnt)
                    data_cnt += 1
                else:
                    print(assay_id)
                    data_cnt += 1

        # pickle.dump(fp_cache, open("/home/fengbin/QSAR/dataset/chembl/chembl_fp.pkl", "wb"))
        train_cnt = len(self.train_indices)
        val_cnt = len(self.val_indices)
        test_cnt = len(self.test_indices)

        self.data_length = {}
        self.data_length['train'] = train_cnt
        self.data_length['val'] = val_cnt
        self.data_length['test'] = test_cnt
        self.data_length['train_weight'] = train_cnt

        print(train_cnt, val_cnt, test_cnt)
        print(np.max([len(x) for x in self.Xs]), np.mean([len(x) for x in self.Xs]))

    def get_split(self, X_in, y_in, scaffold_split=None, **kwargs):
        X, y = X_in, y_in
        split = np.array(scaffold_split)

        if len(X) >= 2048:
            select_idx = [1] * 2048 + [0] * (len(X) - 2048)
            random.shuffle(select_idx)
            select_idx = np.nonzero(np.array(select_idx))
            return [X[select_idx], y[select_idx], split[select_idx]]
        else:
            return [X, y, split]
