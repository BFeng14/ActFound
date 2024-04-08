import math
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
from typing import Dict, List, Set, Tuple, Union
import torch
import random
from collections import defaultdict
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def make_mol(s: str, keep_h: bool, add_h: bool, keep_atom_map: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol


def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False, keep_atom_map=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def preprocess_assay(in_data):
    lines, test_sup_num = in_data
    x_tmp = []
    smiles_list = []
    activity_list = []

    if lines is None:
        return None

    if len(lines) > 10000:
        return None

    for line in lines:
        smiles = line["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
        pic50_exp = line["pic50_exp"]
        activity_list.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(activity_list).astype(np.float32)
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb', 'pqsar', 'fsmol']:
        return None
    return x_tmp, affis, smiles_list


class BaseMetaDataset(Dataset):
    def __init__(self, args, exp_string):
        self.args = args
        self.current_set_name = "train"
        self.exp_string = exp_string

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed,
                          'train_weight': args.train_seed}
        self.batch_size = args.meta_batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.split_all = []

        self.current_epoch = 0
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def get_split(self, X_in, y_in, is_test=False, sup_num=None, scaffold_split=None, y_opls4=None, rand_seed=None,
                  smiles=None):
        def data_split(data_len, sup_num_, rng_):
            if not is_test:
                min_num = math.log10(max(10, int(0.3 * data_len)))
                max_num = math.log10(int(0.85 * data_len))
                # todo:for few-shot setting
                sup_num_ = random.uniform(min_num, max_num)
                sup_num_ = math.floor(10 ** sup_num_)
            split = [1] * sup_num_ + [0] * (data_len - sup_num_)
            rng_.shuffle(split)
            return np.array(split)

        def data_split_byvalue(y, sup_num_):
            sup_index = np.argpartition(y, sup_num_)[:sup_num_].tolist()
            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    split.append(0)
            return np.array(split)

        def data_split_bysaffold(smiles, sup_num_):
            scaffold_dict = scaffold_to_smiles(smiles, use_indices=True)
            scaffold_id_list = [(k, v) for k, v in scaffold_dict.items()]
            scaffold_id_list = sorted(scaffold_id_list, key=lambda x: len(x[1]))
            idx_list_all = []
            for scaffold, idx_list in scaffold_id_list:
                idx_list_all += idx_list

            sup_index = idx_list_all[:sup_num_]
            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    split.append(0)
            return np.array(split)

        def data_split_bysim(Xs, sup_num_, rng_, sim_cut_):
            def get_sim_matrix(a, b):
                a_bool = (a > 0.).float()
                b_bool = (b > 0.).float()
                and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
                or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
                sim = and_res / or_res
                return sim

            Xs_torch = torch.tensor(Xs).cuda()
            sim_matrix = get_sim_matrix(Xs_torch, Xs_torch).cpu().numpy() - np.eye(len(Xs))
            split = [1] * sup_num_ + [0] * (len(Xs) - sup_num_)
            rng_.shuffle(split)
            sup_index = [i for i, t in enumerate(split) if t == 1]

            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    max_sim = np.max(sim_matrix[i][sup_index])
                    if max_sim >= sim_cut_:
                        split.append(-1)
                    else:
                        split.append(0)
            return np.array(split)

        # if np.std(y_in) > 0:
        #     y_in = (np.array(y_in) - np.mean(y_in)) / np.std(y_in)
        # else:
        #     y_in = np.array(y_in) - np.mean(y_in)
        rng = np.random.RandomState(seed=rand_seed)
        if len(X_in) > 64 and not is_test and self.args.datasource != 'gdsc':
            assert y_opls4 is None
            subset_num = 64
            raw_data_len = len(X_in)
            select_idx = [1] * subset_num + [0] * (raw_data_len - subset_num)
            rng.shuffle(select_idx)
            select_idx = np.nonzero(np.array(select_idx))
            X, y = X_in[select_idx], y_in[select_idx]
        else:
            X, y = X_in, y_in

        sup_num = self.args.test_sup_num
        if sup_num <= 1:
            sup_num = sup_num * len(X)
        sup_num = int(sup_num)
        if self.args.similarity_cut < 1.:
            assert is_test
            split = data_split_bysim(X, sup_num, rng, self.args.similarity_cut)
            X = np.array([t for i, t in enumerate(X) if split[i] != -1])
            y = [t for i, t in enumerate(y) if split[i] != -1]
            split = np.array([t for i, t in enumerate(split) if t != -1], dtype=np.int)
        else:
            split = data_split(len(X), sup_num, rng)
        if y_opls4 is not None:
            assert len(y_opls4) == len(y)
            y = (1 - split) * y + split * np.array(y_opls4)

        return [X, y, split]

    def get_set(self, current_set_name, idx):
        datas = []
        assay = []
        si_list = []
        ligand_nums = []
        ligands_all = []

        if current_set_name == 'train':
            si_list = self.train_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ret_weight = [1. for _ in si_list]
        elif current_set_name == 'val':
            si_list = [self.val_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'test':
            si_list = [self.test_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'train_weight':
            if self.idxes is not None:
                si_list = self.idxes[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
                ret_weight = self.train_weight[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
            else:
                si_list = [self.train_indices[idx]]
                ret_weight = [1.]

        for si in si_list:
            ligand_nums.append(len(self.Xs[si]))
            if self.args.expert_test == "fep_opls4":
                y_opls4 = self.assayid2opls4_dict.get(self.assaes[si], None)
            else:
                y_opls4 = None
            assay_name = self.assaes[si]
            if len(self.split_all) > 0:
                scaffold_split = self.split_all[si]
            else:
                scaffold_split = None
            datas.append(self.get_split(self.Xs[si], self.ys[si],
                                        is_test=current_set_name in ['test', 'val', 'train_weight'],
                                        scaffold_split=scaffold_split,
                                        y_opls4=y_opls4,
                                        rand_seed=self.init_seed[current_set_name] + si + self.current_epoch,
                                        smiles=self.smiles_all[si]))
            assay.append(assay_name)
            ligands_all.append(self.smiles_all[si])

        return tuple([[torch.tensor(x[i]) for x in datas] for i in range(0, 3)] +
                     [assay, ret_weight, ligands_all])

    def __len__(self):
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
        self.train_weight = train_weight
        self.idxes = idxes
        self.weighted_batch = weighted_batch

    def switch_set(self, set_name, current_epoch=0):
        self.current_set_name = set_name
        self.current_epoch = current_epoch
        if set_name == "train":
            rng = np.random.RandomState(seed=self.init_seed["train"] + current_epoch)
            rng.shuffle(self.train_indices)

    def __getitem__(self, idx):
        return self.get_set(self.current_set_name, idx=idx)


def my_collate_fn(batch):
    batch = batch[0]
    return batch


class SystemDataLoader(object):
    def __init__(self, args, MetaDataset, current_epoch=0, exp_string=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_epoch: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.args = args
        self.batch_size = args.meta_batch_size
        self.total_train_epochs = 0
        self.dataset = MetaDataset(args, exp_string=exp_string)
        self.full_data_length = self.dataset.data_length
        self.continue_from_epoch(current_epoch=current_epoch)

    def get_train_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=True,
                          collate_fn=my_collate_fn)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    def continue_from_epoch(self, current_epoch):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_epoch:
        """
        self.total_train_epochs += current_epoch

    def get_train_batches_weighted(self, weights=None, idxes=None, weighted_batch=1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train_weight", current_epoch=self.total_train_epochs)
        self.dataset.set_train_weight(weights, idxes, weighted_batch=weighted_batch)
        self.total_train_epochs += 1
        return self.get_dataloader()

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
        self.dataset.switch_set(set_name="train", current_epoch=self.total_train_epochs)
        self.total_train_epochs += self.batch_size
        return self.get_train_dataloader()

    def get_val_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val", current_epoch=repeat_cnt)
        return self.get_dataloader()

    def get_test_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test', current_epoch=repeat_cnt)
        return self.get_dataloader()
