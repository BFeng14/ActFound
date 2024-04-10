import torch as t
import torch.nn as nn
from torch_geometric.nn import GCN, global_add_pool, global_mean_pool, global_max_pool, MLP, GIN, SGConv, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN



class PyGGCN(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGCN, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['GCNInputSize']
        self.hidden_channel = opt.args['GCNHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.num_layers = opt.args['GCNLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor

        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.GCN = GCN(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool
        }
        self.readout = self.ReadoutList[opt.args['GCNReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)
        x = self.GCN(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x

class PyGGIN(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGIN, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['GINInputSize']
        self.hidden_channel = opt.args['GINHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.eps = opt.args['GINEps']
        self.num_layers = opt.args['GINLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor

        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.GIN = GIN(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout,
                       eps = self.eps)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool,
        }
        self.readout = self.ReadoutList[opt.args['GINReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)
        x = self.GIN(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x


class SGC(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return SGConv(in_channels, out_channels, **kwargs)

class PyGSGC(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGSGC, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['SGCInputSize']
        self.hidden_channel = opt.args['SGCHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.K = opt.args['SGCK']
        self.num_layers = opt.args['SGCLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor


        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.SGC = SGC(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout,
                       K = self.K)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool
        }
        self.readout = self.ReadoutList[opt.args['SGCReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)
        x = self.SGC(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x