import numpy as np
import rdkit.Chem as Chem
from TrainingFramework.ChemUtils import GetAtomFeatures, GetBondFeatures
from ogb.utils.features import allowable_features, atom_to_feature_vector, bond_to_feature_vector, safe_index
# five kinds of feature:
# base_one-hot, base-embedding, ogb, rich_one-hot, rich-embedding
def GetNodeFeatureMat(mol):
    FeatureMat = []
    for atom in mol.GetAtoms():
        feature = GetAtomFeatures(atom)
        FeatureMat.append(feature.tolist())
    FeatureMat = np.array(FeatureMat, dtype=np.int64)
    return FeatureMat

def GetBondFeatureMat(mol, bidirection=False):
    FeatureMat = []
    for bond in mol.GetBonds():
        feature = GetBondFeatures(bond)
        FeatureMat.append(feature.tolist())
        if bidirection:
            FeatureMat.append(feature.tolist())
    # mol has no bonds
    if len(FeatureMat) == 0:
        FeatureMat = np.empty((0, 10), dtype=np.int64)
    else:
        FeatureMat = np.array(FeatureMat, dtype=np.int64)
    return FeatureMat

def GetBaseFeatureOH(mol):
    """
    Given a mol, create base feature(one-hot)
    Reference: AttentiveFP/FraGAT
    dimension: atom:39, bond:10
    """
    #list
    x = GetNodeFeatureMat(mol)
    #list
    edge_attr = GetBondFeatureMat(mol, bidirection=True)
    return x, edge_attr


def atom_to_basefeature_vector(atom, feature_list):
    atom_feature = [
        safe_index(feature_list['possible_atomic_symbol'], str(atom.GetSymbol())),
        safe_index(feature_list['possible_degree'], atom.GetDegree()),
        safe_index(feature_list['possible_formal_charge'], atom.GetFormalCharge()),
        safe_index(feature_list['possible_radical_electrons'], atom.GetNumRadicalElectrons()),
        safe_index(feature_list['possible_hybridization'], str(atom.GetHybridization())),
        feature_list['possible_is_aromatic'].index(atom.GetIsAromatic()),
        safe_index(feature_list['possible_num_Implicit_Hs'], atom.GetNumImplicitHs()),
        safe_index(feature_list['possible_chirality'], str(atom.GetChiralTag())),
    ]
    return atom_feature

def bond_to_basefeature_vector(bond, feature_list):
    bond_feature = [
        safe_index(feature_list['possible_bond_type'], str(bond.GetBondType())),
        feature_list['possible_is_conjugated'].index(bond.GetIsConjugated()),
        feature_list['possible_is_in_ring'].index(bond.IsInRing()),
        feature_list['possible_bond_stereo'].index(str(bond.GetStereo())),
    ]
    return bond_feature

def GetBaseFeatureED(mol):
    """
    Given a mol, create base feature(embedding)
    Reference: AttentiveFP/FraGAT
    dimension: atom:8, bond:4
    """
    feature_list = {
        'possible_atomic_symbol': ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','misc'],
        'possible_degree': [0, 1, 2, 3, 4, 5, 'misc'],
        'possible_formal_charge': [-1, -2, 1, 2, 0, 'misc'],
        'possible_radical_electrons': [0, 1, 2, 3, 4, 'misc'],
        'possible_chirality': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER',
            'misc'
        ],
        'possible_num_Implicit_Hs': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
        'possible_is_aromatic': [False, True],
        'possible_is_in_ring_size': [False, True],
        'possible_bond_type': [
            'SINGLE',
            'DOUBLE',
            'TRIPLE',
            'AROMATIC',
            'misc'
        ],
        'possible_bond_stereo': [
            'STEREONONE',
            'STEREOANY',
            'STEREOZ',
            'STEREOE',
            'misc'
        ],
        'possible_is_conjugated': [False, True],
        'possible_is_in_ring': [False, True]
    }
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_basefeature_vector(atom, feature_list))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 4  # bond type, bond stereo, is_conjugated, is_in_ring
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edge_features_list = []
        for bond in mol.GetBonds():
            edge_feature = bond_to_basefeature_vector(bond, feature_list)
            # add edges in both directions
            edge_features_list.append(edge_feature)
            edge_features_list.append(edge_feature)
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
    else:   # mol has no bonds
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return x, edge_attr


def GetOGBFeature(mol):
    """
    Given a mol, create ogb feature based on function 'smiles2graph' in ogb.utils
    atom:9, bond:3
    """
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edge_features_list = []
        for bond in mol.GetBonds():
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edge_features_list.append(edge_feature)
            edge_features_list.append(edge_feature)
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
    else:   # mol has no bonds
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return x, edge_attr


def get_richfeature_atom(atom, ring_info,
                         hydrogen_acceptor_match,
                         hydrogen_donor_match,
                         acidic_match,
                         basic_match):
    """
    Builds a feature vector for an atom.
    """
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
    # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass,
    # 18 for implicitvalence, hydrogen, basic/acidic and ring
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2 + 18
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]
    atom_idx = atom.GetIdx()
    features = features + \
               onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
               [atom_idx in hydrogen_acceptor_match] + \
               [atom_idx in hydrogen_donor_match] + \
               [atom_idx in acidic_match] + \
               [atom_idx in basic_match] + \
               [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                ring_info.IsAtomInRingOfSize(atom_idx, 4),
                ring_info.IsAtomInRingOfSize(atom_idx, 5),
                ring_info.IsAtomInRingOfSize(atom_idx, 6),
                ring_info.IsAtomInRingOfSize(atom_idx, 7),
                ring_info.IsAtomInRingOfSize(atom_idx, 8)]
    assert ATOM_FDIM == len(features)
    return features

def get_richfeature_bond(bond):
    BOND_FDIM = 13
    bt = bond.GetBondType()
    features = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ]
    features += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    assert BOND_FDIM == len(features)
    return features

def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def GetRichFeatureOH(mol):
    """
    Given a mol, create rich feature(one-hot).
    Reference: GROVER
    dimension: atom:151, bond:13
    """
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()

    f_atoms = []
    for _, atom in enumerate(mol.GetAtoms()):
        f_atoms.append(get_richfeature_atom(atom, ring_info, hydrogen_acceptor_match, hydrogen_donor_match,
                                            acidic_match, basic_match))
    f_bonds = []
    for _, bond in enumerate(mol.GetBonds()):
        f_bonds.append(get_richfeature_bond(bond))
        f_bonds.append(get_richfeature_bond(bond))
    if len(f_bonds) == 0:
        f_bonds = np.empty((0, 13), dtype=np.int64)
    else:
        f_bonds = np.array(f_bonds, dtype=np.int64)

    return f_atoms, f_bonds


def atom_to_richfeature_vector(atom, ring_info, hydrogen_acceptor_match,
                               hydrogen_donor_match, acidic_match,
                               basic_match, feature_list):
    atom_idx = atom.GetIdx()
    atom_feature = [
        safe_index(feature_list['possible_atomic_num'], atom.GetAtomicNum()),
        safe_index(feature_list['possible_degree'], atom.GetTotalDegree()),
        safe_index(feature_list['possible_formal_charge'], atom.GetFormalCharge()),
        safe_index(feature_list['possible_chirality'], str(atom.GetChiralTag())),
        safe_index(feature_list['possible_num_Hs'], atom.GetTotalNumHs()),
        safe_index(feature_list['possible_hybridization'], str(atom.GetHybridization())),
        feature_list['possible_is_aromatic'].index(atom.GetIsAromatic()),
        atom.GetMass() * 0.01,
        safe_index(feature_list['possible_implicit_valence'], atom.GetImplicitValence()),
        feature_list['possible_match'].index(atom_idx in hydrogen_donor_match),
        feature_list['possible_match'].index(atom_idx in hydrogen_acceptor_match),
        feature_list['possible_match'].index(atom_idx in acidic_match),
        feature_list['possible_match'].index(atom_idx in basic_match),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 3)),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 4)),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 5)),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 6)),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 7)),
        feature_list['possible_is_in_ring_size'].index(ring_info.IsAtomInRingOfSize(atom_idx, 8)),
        ]
    return atom_feature

def bond_to_richfeature_vector(bond, feature_list):
    bond_feature = [
        safe_index(feature_list['possible_bond_type'], str(bond.GetBondType())),
        feature_list['possible_is_conjugated'].index(bond.GetIsConjugated()),
        feature_list['possible_is_in_ring'].index(bond.IsInRing()),
        safe_index(feature_list['possible_bond_stereo'], str(bond.GetStereo())),
    ]
    return bond_feature

def GetRichFeatureED(mol):
    """
    Given a mol, create rich feature(embedding).
    atom:19, bond:4
    """
    feature_list = {
        'possible_atomic_num': list(range(1, 101)) + ['misc'],
        'possible_degree': [0, 1, 2, 3, 4, 5, 'misc'],
        'possible_formal_charge': [-1, -2, 1, 2, 0, 'misc'],
        'possible_chirality': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_OTHER',
            'misc'
        ],
        'possible_num_Hs': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization': [
            'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
        'possible_is_aromatic': [False, True],
        'possible_implicit_valence': [0, 1, 2, 3, 4, 5, 6, 'misc'],
        'possible_match': [False, True],
        'possible_is_in_ring_size': [False, True],
        'possible_bond_type': [
            'SINGLE',
            'DOUBLE',
            'TRIPLE',
            'AROMATIC',
            'misc'
        ],
        'possible_bond_stereo': [
            'STEREONONE',
            'STEREOZ',
            'STEREOE',
            'STEREOCIS',
            'STEREOTRANS',
            'STEREOANY',
            'misc'
        ],
        'possible_is_conjugated': [False, True],
        'possible_is_in_ring': [False, True]
    }
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
        "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
        "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()

    f_atoms = []
    for _, atom in enumerate(mol.GetAtoms()):
        f_atoms.append(atom_to_richfeature_vector(atom, ring_info, hydrogen_acceptor_match, hydrogen_donor_match,
                                                  acidic_match, basic_match, feature_list))
    f_bonds = []
    for _, bond in enumerate(mol.GetBonds()):
        f_bonds.append(bond_to_richfeature_vector(bond, feature_list))
        f_bonds.append(bond_to_richfeature_vector(bond, feature_list))
    if len(f_bonds) == 0:
        f_bonds = np.empty((0, 4), dtype=np.int64)
    else:
        f_bonds = np.array(f_bonds, dtype=np.int64)

    return f_atoms, f_bonds