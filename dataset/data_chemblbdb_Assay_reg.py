import json
import math
import os
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
            experiment_train = preprocess.read_BDB_per_assay()      
        elif datasource == "chembl":
            experiment_train = preprocess.read_chembl_assay()
        else:
            print("dataset not exist")
            exit()

        self.assay_ids = experiment_train["assays"]

        ligand_set = experiment_train["ligand_sets"]
        self.n_assays = len(ligand_set)

        self.indices = []
        save_dir = '{0}/{1}'.format(self.args.logdir, self.exp_string)
        
        if not os.path.exists(save_dir):
            os.system(f"mkdir -p {save_dir}")

        if datasource == "bdb":
            save_path = '/home/fengbin/dataset/BDB/bdb_split.json'
            self.split_name_train_val_test = json.load(open(save_path, "r"))
            davis_repeat_bdb = list([x.strip() for x in open("/home/fengbin/meta_delta/scripts/cross_repeat/bdb_2_davis_repeat", "r").readlines()])
            print("number of training set before filter:", len(self.split_name_train_val_test['train']))
            fep_repeat_bdb = ["Endothelial-PAS-domain-containing-protein-1/9049_1_1.tsv", "Hepatocyte-growth-factor-receptor/50045505_3_1.tsv",
                                "Cyclin-C/50047223_1_1.tsv", "Poly-[ADP-ribose]-polymerase-tankyrase-2/50007009_4_1.tsv",
                                "Kinesin-like-protein-1/50039105_1_1.tsv", "Mitogen-activated-protein-kinase-8/1994_1_1.tsv",
                                "Mitogen-activated-protein-kinase-14/50033205_13_1.tsv"]
            self.split_name_train_val_test['train'] = [x for x in self.split_name_train_val_test['train'] if
                                                       x not in set(fep_repeat_bdb + davis_repeat_bdb)]
            print("number of training set after filter:", len(self.split_name_train_val_test['train']))
        else:
            save_path = '/home/fengbin/dataset/chembl/chembl_split.json'
            self.split_name_train_val_test = json.load(open(save_path, "r"))

            fep_repeat_chembl = ['CHEMBL3404455_nM_IC50', 'CHEMBL3270296_nM_IC50',
                                 'CHEMBL3779191_nM_IC50', 'CHEMBL4322250_nM_IC50',
                                 'CHEMBL1101849_nM_IC50',  'CHEMBL860181_nM_IC50',
                                 'CHEMBL1769102_nM_IC50', 'CHEMBL2390026_nM_Ki',
                                 'CHEMBL2421816_nM_Ki', 'CHEMBL896126_nM_Ki']
            davis_repeat_chembl = []
            print("number of training set before filter:", len(self.split_name_train_val_test['train']))
            self.split_name_train_val_test['train'] = [x for x in self.split_name_train_val_test['train'] if
                                                      x not in set(fep_repeat_chembl + davis_repeat_chembl)]
            print("number of training set after filter:", len(self.split_name_train_val_test['train']))

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
                print("no expert_test", self.args.expert_test)
                exit()
            ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
            self.split_name_train_val_test['test'] = experiment_test['assays']
        elif self.args.cross_test:
            if datasource == "chembl":
                experiment_test = preprocess.read_bdb_cross()
            else:
                experiment_test = preprocess.read_chembl_cross()
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
