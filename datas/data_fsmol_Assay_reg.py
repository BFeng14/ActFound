import json
import math
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
import concurrent.futures
import pickle
import torch
import preprocess
import copy
import random
import rdkit
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs
# import torchdrug as td
# from torch_geometric.data import HeteroData, Dataset, Batch
from utils.experiment import scaffold_to_smiles, generate_scaffold
from multiprocessing import Pool
from sklearn.model_selection import StratifiedShuffleSplit


def construct_pairs(in_data):
    lines, sim_thres = in_data
    x_tmp = []
    smiles_list = []
    affis = []
    ys_bool = []

    if len(lines) > 10000:
        return None
    for line in lines:
        smiles = line["SMILES"]

        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.RemoveHs(mol)
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
        fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useChirality=True)
        # arr = np.zeros((0,), dtype=np.int8)
        # DataStructs.ConvertToNumpyArray(fp, arr)
        try:
            pic50_exp = eval(line["LogRegressionProperty"])
            ys_bool.append(float(line.get("Property", 0)))
            affis.append(pic50_exp)
            x_tmp.append(fp_numpy)
            smiles_list.append(smiles)
        except:
            pass
            # print(line["LogRegressionProperty"])

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(affis).astype(np.float32)
    ys_bool = np.array(ys_bool)
    if len(x_tmp) < 20:
        return None
    return x_tmp, affis, smiles_list, ys_bool

class CHEMBLMetaDataset(Dataset):
    def __init__(self, args, target_assay, exp_string):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.data_path = args.datadir
        self.experiment_file = args.experiment_file
        self.fp_file = args.fp_file
        self.pair_file = args.pair_file
        self.dataset_name = args.dataset_name
        self.fingerprint_dim = args.dim_w
        self.args = args
        self.train_val_split = args.train_val_split
        self.current_set_name = "train"
        self.exp_string = exp_string
        self.is_test_benchmark = False

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed}
        self.seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed}
        self.batch_size = args.meta_batch_size

        self.num_evaluation_tasks = args.num_evaluation_tasks

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.target_assay = target_assay

        self.rng = np.random.RandomState(seed=self.seed['val'])

        self.load_dataset()

        print("data", self.data_length)
        self.observed_seed_set = None

    def read_assay_type(self, filepath):
        type_file = open(filepath, 'r', encoding='UTF-8', errors='ignore')
        clines = type_file.readlines()
        type_file.close()

        families = {}
        family_type = 0
        assay_types = {}
        for cline in clines:
            cline = str(cline.strip())
            if 'assay_id' not in cline:
                strings = cline.split('\t')
                if strings[3] not in families:
                    families[strings[3]] = family_type
                    family_type += 1
                assay_types[int(strings[0])] = families[strings[3]]

        return assay_types

    def load_dataset(self):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """

        # here only split by training+testing and validation
        # to get the test set, just delete it from either training+testing or validation
        # while in the function
        train_dataset = preprocess.read_fsmol_assay("train", self.args.train)
        valid_dataset = preprocess.read_fsmol_assay("valid")
        test_dataset = preprocess.read_fsmol_assay("test")
        # skin_assay_ids = preprocess.read_skin_assay_ids()
        experiment_test = {}
        # experiment_test = preprocess.read_BDB_merck()
        # experiment_test = preprocess.read_pi3ka()
        # experiment_test = preprocess.read_pi3ka_t47d()
        # experiment_test = preprocess.read_skin_assay()
        # experiment_test = preprocess.read_skin_assay_nodesc("nodesc.CELLVIAB.csv")
        # experiment_test = preprocess.read_skin_assay_nodesc("nodesc.CELLDEATH.csv")
        # experiment_test = preprocess.read_skin_assay_nodesc("nodesc.DERM.csv")
        # experiment_test = preprocess.read_skin_assay_nodesc("nodesc.ATPHGY.csv")
        # experiment_test = preprocess.read_skin_assay_nodesc("nodesc.DERM.logKp.csv")

        self.assay_ids = train_dataset["assays"] + valid_dataset["assays"] + test_dataset["assays"]

        if self.args.use_gnn:
            self.graph_cache = pickle.load(open("/home/fengbin/BindingDB/BDB_assay_pureligand/bdb_mol_graphs.pkl", "rb"))
        else:
            self.graph_cache = {}

        ligand_set = {**train_dataset["ligand_sets"], **valid_dataset["ligand_sets"], **test_dataset["ligand_sets"]}

        self.n_assays = len(ligand_set)

        self.indices = []
        shuffled_assay_ids = copy.deepcopy(self.assay_ids)
        self.rng.shuffle(shuffled_assay_ids)
        save_dir = '{0}/{1}'.format(self.args.logdir, self.exp_string)

        self.split_name_train_val_test = {
            "train": train_dataset["assays"],
            "val": valid_dataset["assays"],
            "test": test_dataset["assays"],
        }

        train_id, val_id, test_id = int(self.train_val_split[0] * self.n_assays), int(
            (self.train_val_split[0] + self.train_val_split[1]) * self.n_assays), int(self.n_assays)

        print(train_id, val_id)

        self.Xs = []
        self.smiles_all = []
        self.ys = []
        self.ys_bool = []
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
        assay_list += self.split_name_train_val_test['val']
        if self.args.train == 1 or self.args.train == 4:
            assay_list += self.split_name_train_val_test['train']

        data_cnt = 0
        with Pool(16) as p:
            res_all = p.map(construct_pairs, tqdm.tqdm([(ligand_set[x], self.args.sim_thres) for x in assay_list]))

            for res, assay_id in zip(res_all, assay_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list, ys_bool = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.ys_bool.append(ys_bool)
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
                    if self.args.scaffold_split:
                        self.test_scaffold_split[data_cnt] = scaffold_to_smiles(mols=smiles_list, use_indices=True)
                    data_cnt += 1
                else:
                    print(assay_id)
                    data_cnt += 1

        self.batch2trainidx = []
        if self.args.train:
            for train_idx in self.train_indices:
                self.batch2trainidx += [train_idx]*1

        # pickle.dump(fp_cache, open("/home/fengbin/QSAR/datas/drug/chembl_fp.pkl", "wb"))
        train_cnt = len(self.train_indices)
        val_cnt = len(self.val_indices)
        test_cnt = len(self.test_indices)

        self.data_length = {}
        self.data_length['train'] = len(self.batch2trainidx)
        self.data_length['val'] = val_cnt
        self.data_length['test'] = test_cnt
        self.data_length['train_weight'] = train_cnt

        print(train_cnt, val_cnt, test_cnt)
        print(np.max([len(x) for x in self.Xs]), np.mean([len(x) for x in self.Xs]))

    def compute_ligand_similarity(self):
        pass

    def compute_smile_intersection(self):
        smiles_set = [[generate_scaffold(smile) for smile in smile_list] for smile_list in self.smiles_all]
        test_sims = []
        for test_assay_idx in self.test_indices:
            test_assay = smiles_set[test_assay_idx]
            intersect_num = []
            for train_assay_idx in self.train_indices:
                train_assay = set(smiles_set[train_assay_idx])
                intersect_num.append(len([x for x in test_assay if x in train_assay]))
            intersect_num = sorted(intersect_num, reverse=True)[:5]
            test_sims.append(np.mean(intersect_num)/len(test_assay))
        print(test_sims)
        return test_sims

    def get_pair_data(self, X_in, y_in, is_test=False, sup_num=None, scaffold_split=None, y_opls4=None):
        def data_split(data_len, sup_num_):
            if not is_test:
                min_num = math.log10(max(10, int(0.3 * data_len)))
                max_num = math.log10(int(0.85 * data_len))
                # todo:for few-shot setting
                sup_num_ = random.uniform(min_num, max_num)
                # sup_num_ = min_num
                sup_num_ = math.floor(10 ** sup_num_)
            split = [1] * sup_num_ + [0] * (data_len - sup_num_)
            random.shuffle(split)
            return np.array(split)

        ## 注意，这里的bug会导致测试的时候，test 样本过少（主要是对于数据特别多的样本）
        if len(X_in) > 512 and not is_test:
            assert y_opls4 is None
            subset_num = 512
            raw_data_len = len(X_in)
            select_idx = [1] * subset_num + [0] * (raw_data_len - subset_num)
            random.shuffle(select_idx)
            select_idx = np.nonzero(np.array(select_idx))
            X, y = X_in[select_idx], y_in[select_idx]
        else:
            X, y = X_in, y_in

        sup_num = self.args.test_sup_num
        if sup_num <= 1:
            sup_num = sup_num * len(X)
        sup_num = int(sup_num)
        split = data_split(len(X), sup_num)
        if y_opls4 is not None:
            assert len(y_opls4) == len(y)
            y = (1-split)*y + split*np.array(y_opls4)

        return [X, y, split]

    def get_set(self, dataset_name, idx):
        datas = []
        assay = []
        si_list = []
        ligand_nums = []
        graphs_all = []

        if dataset_name == 'train':
            si_list = self.batch2trainidx[idx*self.batch_size: (idx+1)*self.batch_size]
            ret_weight = [1. for _ in si_list]
        elif dataset_name == 'val':
            si_list = [self.val_indices[idx]]
            ret_weight = [1.]
        elif dataset_name == 'test':
            si_list = [self.test_indices[idx]]
            ret_weight = [1.]
        elif dataset_name == 'train_weight':
            if self.idxes is not None:
                si_list = self.idxes[idx*self.weighted_batch: (idx+1)*self.weighted_batch]
                ret_weight = self.train_weight[idx*self.weighted_batch: (idx+1)*self.weighted_batch]
            else:
                si_list = [self.train_indices[idx]]
                ret_weight = [1.]

        for si in si_list:
            ligand_nums.append(len(self.Xs[si]))
            if self.args.expert_test == "fep_opls4":
                y_opls4 = self.assayid2opls4_dict[self.assaes[si]]
            else:
                y_opls4 = None
            datas.append(self.get_pair_data(self.Xs[si], self.ys[si], is_test=dataset_name in ['test', 'val', 'train_weight'],
                                            scaffold_split=self.test_scaffold_split.get(si, None), y_opls4=y_opls4))
            assay.append(self.assaes[si])

        return tuple([[torch.tensor(x[i]) for x in datas] for i in range(0, 3)] +
                     [assay, ret_weight])

    def __len__(self):
        if self.is_test_benchmark:
            total_samples = ((len(self.Xs[self.current_assay_idx]) - 10))*20
        else:
            if self.current_set_name == "train":
                total_samples = self.data_length[self.current_set_name] // self.args.meta_batch_size
            elif self.current_set_name == "train_weight":
                if self.idxes is not None:
                    total_samples = len(self.idxes) // self.weighted_batch
                else:
                    total_samples = self.data_length["train_weight"] // self.weighted_batch
            else:
                total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_train_weight(self, train_weight=None, idxes=None, weighted_batch=1):
        self.is_test_benchmark = False
        self.train_weight=train_weight
        self.idxes=idxes
        self.weighted_batch=weighted_batch

    def switch_set(self, set_name, current_iter=0):
        self.is_test_benchmark = False
        self.current_set_name = set_name
        if set_name == "train" :
            rng = np.random.RandomState(seed=self.init_seed["train"] + current_iter)
            rng.shuffle(self.train_indices)

    def switch_assay(self, assay_idx):
        self.is_test_benchmark = True
        self.current_assay_idx = assay_idx

    def update_seed(self, dataset_name, seed=100):
        self.seed[dataset_name] = seed

    def __getitem__(self, idx):
        return self.get_set(self.current_set_name, idx=idx)

    def reset_seed(self):
        self.seed = self.init_seed


def my_collate_fn(batch):
    batch = batch[0]
    return batch


from torch.utils.data.distributed import DistributedSampler
class FsmolMetaLearningSystemDataLoader(object):
    def __init__(self, args, current_iter=0, target_assay=None, exp_string=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_iter: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.args = args
        self.batch_size = args.meta_batch_size
        self.total_train_iters_produced = 0
        self.dataset = CHEMBLMetaDataset(args, target_assay=target_assay, exp_string=exp_string)
        self.full_data_length = self.dataset.data_length
        self.continue_from_iter(current_iter=current_iter)

    def get_train_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        if self.args.ddp:
            return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn,
                          sampler=DistributedSampler(self.dataset))
        else:
            return DataLoader(self.dataset, batch_size=1, num_workers=2, shuffle=True, drop_last=True, collate_fn=my_collate_fn)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    def continue_from_iter(self, current_iter):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_iter:
        """
        self.total_train_iters_produced += (current_iter * (self.batch_size))

    def get_train_batches_weighted(self, weights=None, idxes=None, weighted_batch=1, shuffle=False):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train_weight", current_iter=self.total_train_iters_produced)
        self.dataset.set_train_weight(weights, idxes, weighted_batch=weighted_batch)
        self.total_train_iters_produced += self.batch_size
        if shuffle:
            for sample_id, sample_batched in enumerate(self.get_train_dataloader()):
                yield sample_batched
        else:
            for sample_id, sample_batched in enumerate(self.get_dataloader()):
                yield sample_batched


    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_iter=self.total_train_iters_produced)
        self.total_train_iters_produced += self.batch_size
        return self.get_train_dataloader()

    def get_val_batches(self, total_batches=-1):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val")
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_test_assay_batches(self, assay_idx):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_assay(assay_idx)
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched

    def get_test_batches(self, total_batches=-1):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test')
        for sample_id, sample_batched in enumerate(self.get_dataloader()):
            yield sample_batched
