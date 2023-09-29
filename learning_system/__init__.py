from learning_system.system_meta_delta import MetaDeltaRegressor
from learning_system.system_gdsc import GDSCRegressor
from learning_system.system_protonet import ProtoNetRegressor
from learning_system.system_transfer_delta import TransferDeltaRegressor
from learning_system.system_transfer_qsar import TransferQSARRegressor
from learning_system.system_meta_qsar import MAMLRegressor

def system_selector(model_name):
    if model_name == "meta_delta":
        return MetaDeltaRegressor
    elif model_name == "protonet":
        return ProtoNetRegressor
    elif model_name == "maml":
        return MAMLRegressor
    elif model_name == "transfer_delta":
        return TransferDeltaRegressor
    elif model_name == "transfer_qsar":
        return TransferQSARRegressor
    else:
        raise ValueError(f"model {model_name} is not supported")