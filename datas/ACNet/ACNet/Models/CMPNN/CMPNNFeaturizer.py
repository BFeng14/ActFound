from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
import torch
import numpy as np

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}


# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}


def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim() -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return ATOM_FDIM


def get_bond_fdim() -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    # MolGraph里体现了CMPNN模型中的核心数据结构，即对分子图有向化的扩展

    def __init__(self, smiles, opt):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices  从atom node找入射bond node
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from 从bond node找来源atom node
        self.b2revb = []  # mapping from bond index to the index of the reverse bond  从bond node找反向bond node
        self.bonds = []
        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = mol.GetNumAtoms()

        # Get atom features
        for i, atom in enumerate(mol.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                #print(len(f_bond))
                #print(opt.args['atom_messages'])

                if opt.args['atom_messages']:
                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                else:
                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                #print(len(self.f_atoms[a1] + f_bond))

                # a1, a2是有连边的两个点，连边为bond
                # Update index mappings
                b1 = self.n_bonds  # n_bonds是现在的bond编号到几了？
                b2 = b1 + 1  # b1,b2即为添加的两条边。这里，连边不是通过Mol.GetBonds获得的，而是通过判断任意两个atom之间是否
                # 有连边，来构造的连边，因此这里bond一次构建两条，而且bond的序号和分子中原来的bond的序号无关。
                # 现在有两条边，b1 = a1 --> a2， b2 = a2 --> a1
                self.a2b[a2].append(b1)  # a2b, a2的入射边，增加一个b1
                self.b2a.append(a1)  # b2a，b1的来源点，为a1
                self.a2b[a1].append(b2)  # a2b，a1的入射边，增加一个b2
                self.b2a.append(a2)  # b2a，b2的来源点，为a2
                self.b2revb.append(b2)  # b1的反向边，为b2
                self.b2revb.append(b1)  # b2的反向边，为b1
                self.n_bonds += 2
                self.bonds.append(np.array([a1, a2]))
                # 以上就是构造MolGraph所需要的信息
        # rectify a2b


# =============================================================================
#         for ix in range(len(self.a2b)):
#             if len(self.a2b[ix]) <= 1:
#                 continue
#             if len(self.a2b[ix]) == 2:
#                 self.a2b[ix] = [self.a2b[ix][0], -1, self.a2b[ix][1]]
# =============================================================================
# =============================================================================
#         for ix in range(len(self.a2b)):
#             self.a2b[ix] = sorted(self.a2b[ix])
# =============================================================================

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    # 把一个batch的分子图的数据整合在一起了
    # 用的不是ATFP的方式
    # ATFP是每一个图还是一个图，然后加入pad node，使所有的图的节点数一致
    # 而这里是将所有的节点放在了一起，也就相当于是拼了一个大图。0号节点为pad，1~n1号节点为分子1，n1~n2号节点为分子2，以此类推
    # 整个Batch拼成了一个大图，并通过a_scope和b_scope来识别大图里面的哪些节点是同一个分子的

    def __init__(self, mol_graphs, opt):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + (not opt.args['atom_messages']) * self.atom_fdim  # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        # a_scope和b_scope类似于atom_mask和bond_mask，pad对齐以后告诉模型，哪些是当前分子的真实节点

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features                 # atom features中加入了一个全是0的feature
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features   # bond features中加入了一个全是0的feature
        a2b = [[]]  # mapping from atom index to incoming bond indices    # a2b为空
        b2a = [
            0]  # mapping from bond index to the index of the atom the bond is coming from    # b2a的第一条bond对应的原子编号为0（0是pad）
        b2revb = [
            0]  # mapping from bond index to the index of the reverse bond                 # b2revb的第一条bond0对应的逆向边也是0（自己对自己）
        bonds = [[0, 0]]  # 第0号边，是从0号节点到0号节点的自边。
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)  # atom features中把当前这个molgraph的feature扩展进去了
            f_bonds.extend(mol_graph.f_bonds)  # bond features中把当前这个molgraph的feature扩展进去了
            # 此时，f_atoms的尺寸为[atom_num+1, atom_feature_size]
            # f_bonds同理。其中，第0号均为无用pad

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])  # if b!=-1 else 0
                # b是a2b[a]中的每一个值，也就是入射a的所有bond的编号。将b+self.n_bonds后填入到新的a2b中，
                # 是因为前面加入了pad bond，使得bond的序号整体需要向后移动。初次循环的时候，加入了一条pad bond，因此向后循环1
                # 后面的情况再看。

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1],
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        bonds = np.array(bonds).transpose(1, 0)

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in
                                        a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor(
                [a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(smiles_batch,opt):
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_GRAPH:
            mol_graph = SMILES_TO_GRAPH[smiles]
        else:
            mol_graph = MolGraph(smiles, opt)
            if not opt.args['no_cache']:
                SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, opt)


