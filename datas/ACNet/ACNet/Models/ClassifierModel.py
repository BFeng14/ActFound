import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, opt):
        super(DNN, self).__init__()
        self.output_size = output_size
        self.opt = opt
        self.LayerList = nn.ModuleList()
        if len(layer_sizes) == 0:
            self.FC = nn.Linear(input_size, output_size)
        else:
            for i in range(len(layer_sizes)):
                if i == 0:
                    self.LayerList.append(nn.Linear(input_size, layer_sizes[i]))
                else:
                    self.LayerList.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
                self.LayerList.append(nn.ReLU())
            self.Output = nn.Linear(layer_sizes[-1], output_size)
        self.layer_sizes = layer_sizes
        self.Drop = nn.Dropout(p=self.opt.args['DropRate'])
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(self.layer_sizes) == 0:
            x = self.FC(x)
            if self.opt.args['ClassNum'] != 1:
                if not self.training:
                    # print(f"x size: {x.size()}")
                    x = self.Softmax(x)
        else:
            for layer in self.LayerList:
                x = layer(x)
            x = self.Drop(x)
            x = self.Output(x)
            if self.opt.args['ClassNum'] != 1:
                if not self.training:
                    # print(f"x size: {x.size()}")
                    x = self.Softmax(x)

        return x