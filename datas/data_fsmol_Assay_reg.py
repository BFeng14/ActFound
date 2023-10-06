import json
import math
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
import datas.preprocess as preprocess
import copy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs
from multiprocessing import Pool
from datas.data_base import preprocess_assay, BaseMetaDataset


class FSSMOLMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(FSSMOLMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        train_dataset = preprocess.read_fsmol_assay("train", self.args.train)
        valid_dataset = preprocess.read_fsmol_assay("valid")
        test_dataset = preprocess.read_fsmol_assay("test")
        self.assay_ids = train_dataset["assays"] + valid_dataset["assays"] + test_dataset["assays"]

        ligand_set = {**train_dataset["ligand_sets"], **valid_dataset["ligand_sets"], **test_dataset["ligand_sets"]}

        self.split_name_train_val_test = {
            "train": train_dataset["assays"],
            "val": valid_dataset["assays"],
            "test": test_dataset["assays"],
        }

        self.Xs = []
        self.smiles_all = []
        self.ys = []
        self.assaes = []
        self.sims = []
        self.degrees = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        self.test_scaffold_split = {}

        assay_list = []
        assay_list += self.split_name_train_val_test['test']
        assay_list += self.split_name_train_val_test['val']
        if self.args.train == 1 or self.args.knn_maml:
            assay_list += self.split_name_train_val_test['train']

        data_cnt = 0
        with Pool(16) as p:
            res_all = p.map(preprocess_assay, tqdm.tqdm([(ligand_set[x], self.args.test_sup_num) for x in assay_list]))

            for res, assay_id in zip(res_all, assay_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
                self.assaes.append(assay_id)
                if assay_id in self.split_name_train_val_test['train']:
                    self.train_indices.append(data_cnt)
                    data_cnt += 1
                elif assay_id in self.split_name_train_val_test['val']:
                    self.val_indices.append(data_cnt)
                    data_cnt += 1
                elif assay_id in self.split_name_train_val_test['test']:
                    self.test_indices.append(data_cnt)
                    data_cnt += 1
                else:
                    print(assay_id)
                    data_cnt += 1

        # pickle.dump(fp_cache, open("/home/fengbin/QSAR/datas/chembl/chembl_fp.pkl", "wb"))
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
