from datas.data_gdsc_reg import GDSCMetaDataset
from datas.data_pqsar_Assay_reg import pQSARMetaDataset
from datas.data_fsmol_Assay_reg import FSSMOLMetaDataset
from datas.data_chemblbdb_Assay_reg import CHEMBLBDBMetaDataset
from datas.data_base import SystemDataLoader


def dataset_constructor(args):
    datasource = args.datasource
    if datasource in ["chembl", "bdb"]:
        dataset = CHEMBLBDBMetaDataset
    elif datasource == "fsmol":
        dataset = FSSMOLMetaDataset
    elif datasource == "pqsar":
        dataset = pQSARMetaDataset
    elif datasource == "gdsc":
        dataset = GDSCMetaDataset
    else:
        raise ValueError(f"model {datasource} is not supported")

    return SystemDataLoader(args, dataset)