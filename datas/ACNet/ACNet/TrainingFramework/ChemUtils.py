import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

class ScaffoldGenerator(object):
    # Scaffold generator is used to generate scaffolds for scaffold splitting.
    def __init__(self, include_chirality = False):
        super(ScaffoldGenerator, self).__init__()
        self.include_chirality = include_chirality

    def get_scaffold(self, smiles):
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol = mol,
            includeChirality = self.include_chirality
        )

def GetMol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def GetSmiles(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles

def DrawMolGraph(mol, loc, name):
    AllChem.Compute2DCoords(mol)
    image_name = loc + name + '.png'
    Draw.MolToFile(mol, image_name)

##########################################################
# Functions from RDkit CookBook
##########################################################

def MolGraphWithAtomProp(mol, properties, f=2):
    # Display atom properties in the mol graph
    # f: The number of digits to keep after the decimal point for showing properties
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        property = properties[idx]
        lbl = f'%s:%.{f}f'%(atom.GetSymbol(), property)
        atom.SetProp('atomLabel',lbl)

def MolGraphWithAtomIndex(mol):
    # Display atom index in the mol graph
    properties = []
    for atom in mol.GetAtoms():
        properties.append(atom.GetIdx())
    MolGraphWithAtomProp(mol, properties, f=0)



##########################################################
# Functions to get the topology of the molecular graphs
##########################################################
def GetNeiList(mol):
    atomlist = mol.GetAtoms()
    TotalAtom = len(atomlist)
    NeiList = {}

    for atom in atomlist:
        atomIdx = atom.GetIdx()
        neighbors = atom.GetNeighbors()
        NeiList.update({"{}".format(atomIdx) : []})
        for nei in neighbors:
            neiIdx = nei.GetIdx()
            NeiList["{}".format(atomIdx)].append(neiIdx)

    return NeiList

def GetAdjMat(mol):
    # Get the adjacency Matrix of the given molecule graph
    # If one node i is connected with another node j, then the element aij in the matrix is 1; 0 for otherwise.
    # The type of the bond is not shown in this matrix.

    NeiList = GetNeiList(mol)
    TotalAtom = len(NeiList)
    AdjMat = np.zeros([TotalAtom, TotalAtom])

    for idx in range(TotalAtom):
        neighbors = NeiList["{}".format(idx)]
        for nei in neighbors:
            AdjMat[idx, nei] = 1

    return AdjMat

def GetEdgeList(mol, bidirection=False, offset = 0):
    bondlist = mol.GetBonds()
    edge_list = []
    bond_cnt = 0
    for bond in bondlist:
        bond_idx = bond.GetIdx()
        assert bond_cnt == bond_idx
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        edge_list.append([start_atom+offset, end_atom+offset])
        if bidirection:
            edge_list.append([end_atom+offset, start_atom+offset])
        bond_cnt += 1
    if len(edge_list) == 0:
        edge_list = np.empty((0,2), dtype=np.int64)
    return edge_list


def GetSingleBonds(mol):
    Single_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                bond_idx = bond.GetIdx()
                beginatom = bond.GetBeginAtomIdx()
                endatom = bond.GetEndAtomIdx()
                Single_bonds.append([bond_idx, beginatom, endatom])

    return Single_bonds

def GetCCSingleBonds(mol):
    CCSingle_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            if not bond.IsInRing():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                begin_atom = mol.GetAtomWithIdx(begin_atom_idx).GetSymbol()
                end_atom = mol.GetAtomWithIdx(end_atom_idx).GetSymbol()
                if (begin_atom == 'C') & (end_atom == 'C'):
                    bond_idx = bond.GetIdx()
                    CCSingle_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
    return CCSingle_bonds

def GetCXSingleBonds(mol):
    '''
    Proposal:
    1. Get the acyclic CC single bond
    2. Get the acyclic single bond between heteroatoms and acyclic carbon(sp3 hybridization,excluding quaternary carbon)
    3. Get the acyclic single bond between heteroatoms and carbon connected to a ring
    '''
    CXSingle_bonds = list()
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            # the bond is not in a ring.
            if not bond.IsInRing():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                begin_atom = mol.GetAtomWithIdx(begin_atom_idx).GetSymbol()
                end_atom = mol.GetAtomWithIdx(end_atom_idx).GetSymbol()
                ba_hybrid = mol.GetAtomWithIdx(begin_atom_idx).GetHybridization()
                ba_ring = mol.GetAtomWithIdx(begin_atom_idx).IsInRing()
                ba_hydrogen = mol.GetAtomWithIdx(begin_atom_idx).GetTotalNumHs()
                ea_hybrid = mol.GetAtomWithIdx(end_atom_idx).GetHybridization()
                ea_ring = mol.GetAtomWithIdx(end_atom_idx).IsInRing()
                ea_hydrogen = mol.GetAtomWithIdx(end_atom_idx).GetTotalNumHs()
                # C-C single bonds
                if (begin_atom == 'C') & (end_atom == 'C'):
                    bond_idx = bond.GetIdx()
                    CXSingle_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
                # C-X single bonds
                # the hybridization type of carbon is sp3, except that the carbon is not connected to any hydrogen atom.
                # the carbon is not in a ring.
                elif ((begin_atom == 'C') \
                      & (not(ba_ring)) \
                      & (ba_hybrid == 'SP3') \
                      & (ba_hydrogen != 0) \
                      & (end_atom != 'C')) or \
                    ((end_atom == 'C') \
                      & (not(ea_ring)) \
                      & (ea_hybrid == 'SP3') \
                      & (ea_hydrogen != 0) \
                      & (begin_atom != 'C')):
                    bond_idx = bond.GetIdx()
                    CXSingle_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
                # C-X single bonds
                # the carbon is in a ring.
                elif ((begin_atom == 'C') \
                      & (ba_ring) \
                      & (end_atom != 'C')) or \
                     ((end_atom == 'C') \
                      & (ea_ring) \
                      & (begin_atom !='C')):
                    bond_idx = bond.GetIdx()
                    CXSingle_bonds.append([bond_idx, begin_atom_idx, end_atom_idx])
    return CXSingle_bonds
##########################################################
# Functions to get the chemical features of the atoms and the bonds
##########################################################
def GetAtomFeatures(atom):
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At']
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc+24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31+hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[36] = 1
        try:
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R','S']
            loc = ChiList.index(chi)
            feature[37+loc] = 1
            #print("Chirality resolving finished.")
        except:
            feature[37] = 0
            feature[38] = 0
    return feature

def GetBondFeatures(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    BondTypeList = [rdkit.Chem.rdchem.BondType.SINGLE,
                    rdkit.Chem.rdchem.BondType.DOUBLE,
                    rdkit.Chem.rdchem.BondType.TRIPLE,
                    rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in BondTypeList:
        loc = BondTypeList.index(type)
        feature[0+loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = conj

    # stereo
    stereo = bond.GetStereo()
    StereoList = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                  rdkit.Chem.rdchem.BondStereo.STEREOANY,
                  rdkit.Chem.rdchem.BondStereo.STEREOZ,
                  rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in StereoList:
        loc = StereoList.index(stereo)
        feature[6+loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature

def GetMolFeatureMat(mol):
    FeatureMat = []
    for atom in mol.GetAtoms():
        feature = GetAtomFeatures(atom)
        FeatureMat.append(feature.tolist())
    return FeatureMat

def GetNodeFeatureMat(mol):
    NodeFeatureMat = []
    for atom in mol.GetAtoms():
        node_feature = GetAtomFeatures(atom)
        NodeFeatureMat.append(node_feature.tolist())
    return NodeFeatureMat

def GetBondFeatureMat(mol, bidirection=False):
    FeatureMat = []
    for bond in mol.GetBonds():
        feature = GetBondFeatures(bond)
        FeatureMat.append(feature.tolist())
        if bidirection:
            FeatureMat.append(feature.tolist())

    # mol has no bonds
    if len(FeatureMat) == 0:
        FeatureMat = np.empty((0, 10), dtype = np.int64)
    else:
        FeatureMat = np.array(FeatureMat, dtype = np.int64)
    return FeatureMat

############################################################
# Calculator to get fingerprints of a given molecule
############################################################
class BasicMolFPCalculator(object):
    def __init__(self):
        super(BasicMolFPCalculator, self).__init__()
    def _calculate_fingerprint(self,mol,opt_array=None):
        raise NotImplementedError(
            "Fingerprint calculating method not implemented."
        )
    def CalculateFP(self,mol,opt_array=None):
        FP = self._calculate_fingerprint(mol,opt_array)
        return FP

class MorganFPMolFPCalculator(BasicMolFPCalculator):
    # To calculate the Morgan FP of a given molecule.
    def __init__(self):
        super(MorganFPMolFPCalculator, self).__init__()

    def _calculate_fingerprint(self,mol,opt_array):
        if opt_array==None:
            raise TypeError(
                'opt_array is not given.'
            )

        if 'nBits' not in opt_array:
            raise KeyError(
                'nBits of the FP not given.'
            )
        if 'radius' not in opt_array:
            raise KeyError(
                'radius of the FP not given.'
            )

        nBits = opt_array['nBits']
        radius = opt_array['radius']

        if 'ToBit' in opt_array:
            ToBit = opt_array['ToBit']
        else:
            ToBit = False

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

        if ToBit:
            FP = fp.ToBitString()
            FP_array = []
            for i in range(len(FP)):
                FP_value = float(FP[i])
                FP_array.append(FP_value)
            return FP_array
        else:
            return fp

class RDKFPMolFPCalculator(BasicMolFPCalculator):
    # To calculate the RDK Fingerprint of a given molecule
    # also known as Topological Fingerprint
    def __init__(self):
        super(RDKFPMolFPCalculator, self).__init__()

    def _calculate_fingerprint(self,mol,opt_array=None):
        if opt_array != None:
            if 'ToBit' in opt_array:
                ToBit = opt_array['ToBit']
            else:
                ToBit = False
        else:
            ToBit = False

        fp = Chem.RDKFingerprint(mol)

        if ToBit:
            FP = fp.ToBitString()
            FP_array = []
            for i in range(len(FP)):
                FP_value = float(FP[i])
                FP_array.append(FP_value)
            return FP_array
        else:
            return fp

class MACCSFPMolFPCalculator(BasicMolFPCalculator):
    def __init__(self):
        super(MACCSFPMolFPCalculator, self).__init__()

    def _calculate_fingerprint(self,mol,opt_array=None):
        if opt_array != None:
            if 'ToBit' in opt_array:
                ToBit = opt_array['ToBit']
            else:
                ToBit = False
        else:
            ToBit = False

        fp = MACCSkeys.GenMACCSKeys(mol)

        if ToBit:
            FP = fp.ToBitString()
            FP_array = []
            for i in range(len(FP)):
                FP_value = float(FP[i])
                FP_array.append(FP_value)
            return FP_array
        else:
            return fp


##########################################################
# Calculator to get similarity between molecules
##########################################################
class MolSimCalculator(object):
    def __init__(self, opt):
        super(MolSimCalculator, self).__init__()
        self.opt = opt

        self.FPCalculators = {
            'RDKFP': RDKFPMolFPCalculator(),
            'MorganFP': MorganFPMolFPCalculator(),
            'MACCSFP': MACCSFPMolFPCalculator()
        }
        '''
        self.MetricFuncs = {
            'Tanimoto': DataStructs.TanimotoSimilarity,
            'Dice': DataStructs.DiceSimilarity,
            'Cosine': DataStructs.CosineSimilarity
        } # These metrics funcs cannot be pickle so that it cannot be switched by opt when using multi-processing methods.
        '''
        self.FPcalculator = self.FPCalculators[self.opt.args['SimNetFP']]
        #self.metricfunc = self.MetricFuncs[opt['similarity']]
        if self.opt.args['SimNetFP'] == 'MorganFP':
            if 'radius' not in self.opt.args:
                raise KeyError(
                    'radius of the FP not given.'
                )
            elif 'nBits' not in self.opt.args:
                raise KeyError(
                    'nBits of the FP not given.'
                )
            else:
                self.FP_opt_array = {
                    'radius': self.opt.args['radius'],
                    'nBits': self.opt.args['nBits']
                }
        else:
            self.FP_opt_array = {}

    def CalculateSim(self,mol1,mol2):
        FP1 = self.FPcalculator.CalculateFP(mol1, self.FP_opt_array)
        FP2 = self.FPcalculator.CalculateFP(mol2, self.FP_opt_array)
        sim = DataStructs.FingerprintSimilarity(FP1,FP2)
        # Using Tanimoto Similarity as default, because other similarity calculating functions cannot be pickled
        #sim = DataStructs.FingerprintSimilarity(FP1,FP2,self.metricfunc)
        return sim

##########################################################
# Checkers to screen the dataset
##########################################################
class BasicChecker(object):
    def __init__(self):
        super(BasicChecker, self).__init__()

    def check(self, dataset):
        raise NotImplementedError(
            "Dataset Checker not implemented.")

class MolChecker(BasicChecker):
    def __init__(self):
        super(MolChecker, self).__init__()

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = GetMol(smiles)
            if mol:
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


class AttentiveFPChecker(BasicChecker):
    # Rules proposed in the source code of Attentive FP
    # To screen the samples that not satisfy the rules
    # more rules can be added.
    def __init__(self, max_atom_num, max_degree):
        super(AttentiveFPChecker, self).__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
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




#########################################################
# Activity Cliff Screener to screen whether ACs exists between mol pair
#########################################################
class ActivityCliffScreener(object):
    def __init__(self, opt):
        super(ActivityCliffScreener, self).__init__()
        self.opt = opt

        self.MolPairRules = {'Similarity': self.SimilarityMolPairCheck(),
                             'MSN': self.MSNCheck(),
                        }
        self.ActivityCliffRules = {'Binary': self.BinaryActivityCliffCheck(),}

        self.MolPairChecker = self.MolPairRules[self.opt.args['MolPairRule']]
        self.ActivityCliffChecker = self.ActivityCliffRules[self.opt.args['ActivityCliffRule']]

    #def CheckAC(self, mol1, mol2):


    def SimilarityMolPairCheck(self, mol1, mol2):
        print("Not Implemented")
        return

    #def MSN(self, mol1, mol2):
