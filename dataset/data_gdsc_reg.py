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
from multiprocessing import Pool
from dataset.data_base import preprocess_assay, BaseMetaDataset


class GDSCMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(GDSCMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        assert self.args.datasource == "gdsc"
        experiment_train = preprocess.read_gdsc()
        self.assay_ids = experiment_train["assays"]
        self.split_name_train_val_test = {
            "train": self.assay_ids[:100],
            "valid": [],
            "test": self.assay_ids[100:],
        }

        ligand_set = experiment_train["ligand_sets"]
        self.n_assays = len(ligand_set)

        self.indices = []
        save_dir = '{0}/{1}'.format(self.args.logdir, self.exp_string)
        
        if not os.path.exists(save_dir):
            os.system(f"mkdir -p {save_dir}")

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

        ma_pair = 0
        assay_list = []
            
        assay_list += self.split_name_train_val_test['test']
        # valid set
        assay_list += self.split_name_train_val_test['valid']
        # train set
        if self.args.train == 1 or self.args.train == 4:
            assay_list += self.split_name_train_val_test['train']

        data_cnt = 0
        cellline_feats = pickle.load(open(self.args.cell_line_feat, "rb"))
        with Pool(16) as p:
            res_all = p.map(preprocess_assay, tqdm.tqdm([(ligand_set[x], self.args.test_sup_num) for x in assay_list]))

            for res, assay_id in zip(res_all, assay_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list = res
                if assay_id not in cellline_feats.keys():
                    continue

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
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
        # self.args.datasource = "bdb"
