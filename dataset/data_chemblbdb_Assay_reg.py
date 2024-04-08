import json
import math
import os
import random

import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
import pickle
import torch

import dataset.load_dataset as preprocess
import copy
from multiprocessing import Pool
from dataset.data_base import preprocess_assay, BaseMetaDataset


class CHEMBLBDBMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(CHEMBLBDBMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        datasource = self.args.datasource

        if datasource == "bdb":
            experiment_train = preprocess.read_BDB_per_assay(self.args)
        elif datasource == "chembl":
            experiment_train = preprocess.read_chembl_assay(self.args)
        elif datasource == "bdb_ic50":
            experiment_train = preprocess.read_BDB_IC50()
        else:
            print("dataset not exist")
            exit()

        self.assay_ids = experiment_train["assays"]

        ligand_set = experiment_train["ligand_sets"]
        self.n_assays = len(ligand_set)

        print(len(ligand_set))

        self.indices = []
        save_dir = '{0}/{1}'.format(self.args.logdir, self.exp_string)
        
        if not os.path.exists(save_dir):
            os.system(f"mkdir -p {save_dir}")

        if datasource == "bdb":
            save_path = f'{preprocess.DATA_PATH}/BDB/bdb_split.json'
            self.split_name_train_val_test = json.load(open(save_path, "r"))
            davis_repeat_bdb = json.load(open(f"{preprocess.DATA_PATH}/davis_repeat_set_on_bdb_0.95.json", "r"))
            fep_repeat_bdb = json.load(open(f"{preprocess.DATA_PATH}/fep_repeat_set_on_bdb_0.95.json", "r"))
            activity_repeat_bdb = json.load(open(f"{preprocess.DATA_PATH}/activity_cliff_repeat_set_on_bdb_nomid_0.95.json", "r"))

            repea_set_bdb = set(davis_repeat_bdb + fep_repeat_bdb + activity_repeat_bdb)
            print("number of training set before filter:", len(self.split_name_train_val_test['train']))
            self.split_name_train_val_test['train'] = [x for x in self.split_name_train_val_test['train'] if
                                                       x not in repea_set_bdb]
            print("number of training set after filter:", len(self.split_name_train_val_test['train']))
        elif datasource == "chembl":
            save_path = f'{preprocess.DATA_PATH}/chembl/chembl_split.json'
            self.split_name_train_val_test = json.load(open(save_path, "r"))
            davis_repeat_chembl = json.load(open(f"{preprocess.DATA_PATH}/davis_repeat_set_on_chembl_0.95.json", "r"))
            fep_repeat_chembl = json.load(open(f"{preprocess.DATA_PATH}/fep_repeat_set_on_chembl_0.95.json", "r"))
            activity_repeat_chembl = json.load(open(f"{preprocess.DATA_PATH}/activity_cliff_repeat_set_on_chembl_nomid_0.95.json", "r"))

            repea_set_chembl = set(davis_repeat_chembl + fep_repeat_chembl + activity_repeat_chembl)
            print("number of training set before filter:", len(self.split_name_train_val_test['train']))
            self.split_name_train_val_test['train'] = [x for x in self.split_name_train_val_test['train'] if
                                                      x not in repea_set_chembl]
            print("number of training set after filter:", len(self.split_name_train_val_test['train']))
        elif datasource == "bdb_ic50":
            self.split_name_train_val_test = {}
            assays_name_list = experiment_train["assays"]
            self.split_name_train_val_test['train'] = [x for x in assays_name_list if not x.startswith('test_')]
            self.split_name_train_val_test['valid'] = [x for x in assays_name_list if x.startswith('test_')]
            self.split_name_train_val_test['test'] = []


        self.Xs = []
        self.smiles_all = []
        self.ys = []
        self.assaes = []
        self.sims = []
        self.degrees = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        assay_list = []
        # test set
        if self.args.expert_test != "":
            if self.args.expert_test == "fep":
                experiment_test, _ = preprocess.read_FEP_SET()
            elif self.args.expert_test == "fep_opls4":
                experiment_test, self.assayid2opls4_dict = preprocess.read_FEP_SET()
            elif self.args.expert_test == "kiba":
                experiment_test = preprocess.read_kiba()
            elif self.args.expert_test == "davis":
                experiment_test = preprocess.read_davis()
            elif self.args.expert_test == "ood":
                experiment_test = preprocess.read_chembl_cell_assay_OOD()
            else:
                raise ValueError("no expert_test", self.args.expert_test)
            ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
            self.split_name_train_val_test['test'] = experiment_test['assays']
        elif self.args.cross_test:
            if datasource == "chembl":
                experiment_test = preprocess.read_bdb_cross(self.args)
            else:
                experiment_test = preprocess.read_chembl_cross(self.args)
            ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
            self.split_name_train_val_test['test'] = experiment_test['assays']

        assay_list += self.split_name_train_val_test['test']
        # valid set
        assay_list += self.split_name_train_val_test['valid']
        # train set
        if self.args.train == 1 or self.args.knn_maml:
            assay_list += self.split_name_train_val_test['train']

        data_cnt = 0
        with Pool(16) as p:
            res_all = p.map(preprocess_assay, tqdm.tqdm([(ligand_set.get(x, None), self.args.test_sup_num) for x in assay_list]))

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
                    repeat_cnt = len(smiles_list) // 64
                    for i in range(repeat_cnt):
                        self.Xs.append(x_tmp)
                        self.ys.append(y_tmp)
                        self.smiles_all.append(smiles_list)
                        self.assaes.append(f"{assay_id}_{i}")
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
