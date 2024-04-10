import torch as t
import torch.nn as nn

class Initializer(object):
    def __init__(self):
        super(Initializer, self).__init__()

    def WeightInit(self, tensor):
        self._init_func(tensor)

    def _init_func(self, tensor):
        raise NotImplementedError("Weight Initialization Function is not implemented.")


class NormalInitializer(Initializer):
    def __init__(self, opt):
        self.opt = opt
        super(NormalInitializer, self).__init__()

    def _init_func(self, tensor):
        mean = self.opt.args['InitMean']
        std = self.opt.args['InitStd']
        nn.init.normal_(tensor, mean, std)


class XavierNormalInitializer(Initializer):
    def __init__(self):
        super(XavierNormalInitializer, self).__init__()

    def _init_func(self, tensor):
        if tensor.dim() == 1:
            nn.init.constant_(tensor, 0)
        else:
            nn.init.xavier_normal_(tensor)