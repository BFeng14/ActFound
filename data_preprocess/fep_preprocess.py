import sys
import os
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
absolute_path = os.path.abspath(__file__)
DATA_PATH = "/" + "/".join(absolute_path.split("/")[:-2]+["datas"])

def remove_hs(d3o_mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    d3o_mol_no_h = Chem.RemoveHs(d3o_mol, params)

    uncharger = rdMolStandardize.Uncharger()
    uncharged_d3o_mol = uncharger.uncharge(d3o_mol_no_h)
    deuterated_uncharged_d3o_mol = Chem.RemoveHs(Chem.AddHs(uncharged_d3o_mol))
    return deuterated_uncharged_d3o_mol


def converter(file_name):
    mols = [mol for mol in Chem.SDMolSupplier(file_name)]
    convert_dict = {}
    for mol in mols:
        name = mol.GetProp("_Name")
        mol = remove_hs(mol)
        smi = Chem.MolToSmiles(mol)
        inchi = Chem.MolToInchi(mol)#MolStandardize.rdMolStandardize.Uncharger.uncharge()(mol)
        convert_dict[name] = (smi, inchi)
    return convert_dict


result_dir = f"{DATA_PATH}/FEP/public_BFE_benchmark/results"
data_dict = {}
for file in os.listdir(result_dir):
    assay_name = file.split("_")[0]
    sdf_file = f"{DATA_PATH}/FEP/public_BFE_benchmark/sdfs/{assay_name}_ligands.sdf"
    convert_dict = converter(sdf_file)
    inchi_set = set()
    ligands = []
    for line in list(open(os.path.join(result_dir, file)))[1:]:
        line = line.strip().split(",")
        name = line[0]
        if assay_name == "mcl1":
            exp_dg = line[3]
            fep_dg = line[5]
        else:
            exp_dg = line[1]
            fep_dg = line[2]
        smi, inchi_key = convert_dict[name]
        if inchi_key not in inchi_set:
            inchi_set.add(inchi_key)
        else:
            print(smi)
            continue
        ligands.append({
            "name":name,
            "smiles": smi,
            "exp_dg": exp_dg,
            "pred_dg": fep_dg
        })
    data_dict[assay_name] = ligands

import json
json.dump(data_dict, open(f"{DATA_PATH}/FEP/fep_data_final_norepeat_nocharge.json", "w"))