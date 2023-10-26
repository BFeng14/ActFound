from learning_system.system_actfound import ActFoundRegressor
from learning_system.system_gdsc import GDSCRegressor
from learning_system.system_protonet import ProtoNetRegressor
from learning_system.system_actfound_transfer import ActFoundTransferRegressor
from learning_system.system_transfer_qsar import TransferQSARRegressor
from learning_system.system_meta_qsar import MAMLRegressor

def system_selector(model_name):
    if model_name == "actfound":
        return ActFoundRegressor
    elif model_name == "protonet":
        return ProtoNetRegressor
    elif model_name == "maml":
        return MAMLRegressor
    elif model_name == "actfound_transfer":
        return ActFoundTransferRegressor
    elif model_name == "transfer_qsar":
        return TransferQSARRegressor
    else:
        raise ValueError(f"model {model_name} is not supported")