import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
from TrainingFramework.ChemUtils import BasicChecker, GetMol

class ACMolChecker(BasicChecker):
    def __init__(self, pair_wise=None):
        super(ACMolChecker, self).__init__()
        self.pair_wise = pair_wise

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            if not self.pair_wise:
                smiles = item['SMILES']
                mol = GetMol(smiles)
                if mol:
                    checked_dataset.append(item)
                else:
                    discarded_dataset.append(item)
            else:
                smiles1 = item['SMILES1']
                smiles2 = item['SMILES2']
                mol1 = GetMol(smiles1)
                mol2 = GetMol(smiles2)
                if mol1 and mol2:
                    checked_dataset.append(item)
                else:
                    discarded_dataset.append(item)
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules have passed check.")
        print(len(discarded_dataset), " molecules have been discarded.")
        print("Discarded molecules:")
        print(discarded_dataset)
        return checked_dataset

class ACAttentiveFPChecker(BasicChecker):
    # Rules proposed in the source code of Attentive FP
    # To screen the samples that not satisfy the rules
    # more rules can be added.
    def __init__(self, max_atom_num, max_degree, pair_wise=None):
        super(ACAttentiveFPChecker, self).__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0
        self.pair_wise = pair_wise

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            if self.pair_wise:
                smiles1 = item['SMILES1']
                smiles2 = item['SMILES2']
                mol1 = GetMol(smiles1)
                mol2 = GetMol(smiles2)
                if mol1 and mol2:
                    #self.check_single_bonds(mol)
                    self.check_degree(mol1)
                    self.check_degree(mol2)
                    self.check_max_atom_num(mol1)
                    self.check_max_atom_num(mol2)
                    if self.mol_error_flag == 0:
                        checked_dataset.append(item)
                    else:
                        discarded_dataset.append(item)
                        self.mol_error_flag = 0
                else:
                    discarded_dataset.append(item)
                    self.mol_error_flag = 0
            else:
                smiles = item['SMILES']
                mol = GetMol(smiles)
                #check
                if mol:
                    #self.check_single_bonds(mol)
                    self.check_degree(mol)
                    self.check_max_atom_num(mol)
                    if self.mol_error_flag == 0:
                        checked_dataset.append(item)
                    else:
                        discarded_dataset.append(item)
                        self.mol_error_flag = 0
                else:
                    discarded_dataset.append(item)
                    self.mol_error_flag = 0

        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")
        print("Discarded molecules:")
        print(discarded_dataset)
        return checked_dataset

    def check_degree(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetDegree() > self.max_degree:
                self.mol_error_flag = 1
                break

    def check_max_atom_num(self, mol):
        if len(mol.GetAtoms()) > self.max_atom_num:
            self.mol_error_flag = 1

    def check_single_bonds(self, mol):
        # check whether there is at least one single bond in the molecule
        # this check is not used in FraGAT
        self.mol_error_flag = 1
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if not bond.IsInRing():
                    self.mol_error_flag = 0
                    break