from learning_system.system_actfound import ActFoundRegressor
from learning_system.system_gdsc import GDSCRegressor
from learning_system.system_protonet import ProtoNetRegressor
from learning_system.system_actfound_transfer import ActFoundTransferRegressor
from learning_system.system_transfer_qsar import TransferQSARRegressor
from learning_system.system_meta_qsar import MAMLRegressor


def system_selector(args):
    model_name = args.model_name
    if args.datasource == "gdsc":
        return GDSCRegressor
    if model_name == "actfound":
        return ActFoundRegressor
    if model_name == "protonet":
        return ProtoNetRegressor
    if model_name == "maml":
        return MAMLRegressor
    if model_name == "actfound_transfer":
        return ActFoundTransferRegressor
    if model_name == "transfer_qsar":
        return TransferQSARRegressor

    raise ValueError(f"model {model_name} is not supported")