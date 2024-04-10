import torch as t
import torch.nn as nn
import torch.nn.functional as F
from Models.CMPNN.nn_utils import get_activation_function, index_select_ND
from Models.CMPNN.CMPNNFeaturizer import mol2graph,get_atom_fdim, get_bond_fdim
import math

class CommunicateKernel(nn.Module):
    def __init__(self, opt):
        super(CommunicateKernel, self).__init__()
        self.opt = opt
        self.kernel = self.opt.args['CommunicateKernel']

        # if self.kernel == 'MultilayerPerception':
        #     self.linear = nn.Linear()

    def forward(self, hidden, agg_message):
        # hidden: h^{k-1} (v)
        # agg_message: m^k (v)

        if self.kernel == 'Add':
            return hidden + agg_message
        # elif self.opt.args['CommunicateKernel'] == 'MultilayerPerception':

class MPNLayer(nn.Module):
    def __init__(self, opt):
        super(MPNLayer, self).__init__()
        self.opt = opt
        self.hidden_size = self.opt.args['FPSize']
        self.W_bond = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=self.opt.args['DropRate'])
        self.act_func = get_activation_function(opt)
        self.communicate_kernel = CommunicateKernel(opt)

    def forward(self, message_atom, message_bond, a2b, b2a, b2revb, input_bond):
        # message_atom: h^{k-1} (v)
        # message_bond: h^{k-1} (e_{wv})
        # a2b, b2a, b2revb are index to find neighbors
        # input_bond: h^0 (e_{vw})
        # nodes
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = self.MessageBooster(agg_message)
        message_atom = self.communicate_kernel(message_atom, agg_message)

        # edges
        rev_message = message_bond[b2revb]
        message_bond = message_atom[b2a] - rev_message
        message_bond = self.W_bond(message_bond)
        message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

        return message_atom, message_bond

    def MessageBooster(self, agg_message):
        return agg_message.sum(dim=1) * agg_message.max(dim=1)[0]

class BatchGRU(nn.Module):
    def __init__(self, hidden_size):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first = True,
                          bidirectional = True)
        self.bias = nn.Parameter(t.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        # 输入：node为一个batch的大图中所有节点v的features， a_scope为这个batch的大图中，哪些节点隶属于一个mol
        #
        hidden = node
        message = F.relu(node + self.bias)  # 节点信息加了一个偏置以后过relu激活（线性系数为1）
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])  # 最大的原子数量
        # padding
        message_lst = []
        hidden_lst = []

        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            # torch.Tensor.narrow函数的功能是从第dimension维中，从start开始，选取length个，得到切片
            cur_hidden = hidden.narrow(0, a_start, a_size)
            # message和hidden的区别：hidden是K层以后得到的各个节点的feature，message是加了偏置并激活以后的feature

            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            # cur_hidden的尺寸应该是[a_size, hidden_size]
            # cur_hidden.max(0)[0]的结果是返回cur_hidden中，feature各个元素在不同atom上的最大值，返回尺寸[hidden_size]
            # 两次unsquezze(0)以后的尺寸为[1,1,hidden_size]，append到list中

            cur_message = t.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            # 这句话就是简单的填充。把所有的cur_message，按照最大原子数填充一致
            # 从[a_size, hidden_size]填充为[max_atom_len, hidden_size]
            message_lst.append(cur_message.unsqueeze(0))
            # unsqueeze成[1,max_atom_len,hidden_size]后，append到list中

        message_lst = t.cat(message_lst, 0)
        hidden_lst = t.cat(hidden_lst, 1)
        # 把两个list转化为两个tensor。list的长度均为batch_size
        # message_lst的尺寸为[batch_size, max_atom_len, hidden_size]
        # hidden_lst的尺寸为[1,batch_size, hidden_size]
        hidden_lst = hidden_lst.repeat(2, 1, 1)  # [2,batch_size,hidden_size]
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        # 这里，GRU的输入，input为message_lst，h_0为hidden_lst。也就是说，K层MPN提出来的node embedding是GRU的初始隐变量
        # 而message_lst则是序列化的输入特征
        # 这里gru只有一层,但是因为选择了bidirectional，所以要把h复制成2.

        # 所以可以明确，GRU的作用就是，将一个分子中的各个原子的feature，按照序列输入到GRU，GRU将融合这个序列前后的其它各个原子的信息，对这个原子的信息进行更新
        # 这个GRU接收的序列，是一个分子中的各个原子组成的序列，因此并不是"不同层的信息更新"，也没有包含有拓扑信息，
        # 因为这个序列只是按照原子序号进行组合的，并没有考虑拓扑
        # 所以这个GRU的作用值得商榷

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = t.cat(cur_message_unpadding, 0)

        message = t.cat([t.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


#######################################################################################

class MPNEncoder(nn.Module):
    def __init__(self, opt, atom_fdim, bond_fdim):
        super(MPNEncoder, self).__init__()
        self.opt = opt
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = opt.args['FPSize']
        #self.bias = opt.args['bias']
        self.depth = opt.args['CMPNNLayers']

        # print(f"atom dim:{self.atom_fdim}")
        # print(f"hidden size: {self.hidden_size}")
        self.W_i_atom = nn.Linear(self.atom_fdim, self.hidden_size)
        self.W_i_bond = nn.Linear(self.bond_fdim, self.hidden_size)

        self.MPNLayers = nn.ModuleList()
        for k in range(self.depth - 1):
            self.MPNLayers.append(MPNLayer(opt))

        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.gru = BatchGRU(hidden_size = self.hidden_size)

        self.W_o = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.act_func = get_activation_function(opt)
        self.dropout_layer = nn.Dropout(p=self.opt.args['DropRate'])


    def forward(self, input):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = self._unpack_inputs(input)

        # Input feature transform
        #print(f_atoms[0])
        #print(f_atoms.size())
        input_atom = self.W_i_atom(f_atoms)
        #print(input_atom.size())
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        #print(f_bonds)
        #print(f_bonds.size())
        #print(self.W_i_bond)
        input_bond = self.W_i_bond(f_bonds)
        input_bond = self.act_func(input_bond)
        message_bond = input_bond.clone()

        # Message Passing
        for layer in self.MPNLayers:
            message_atom, message_bond = layer(message_atom, message_bond, a2b,b2a,b2revb,input_bond)

        agg_message = index_select_ND(message_bond, a2b)
        agg_message = self.MessageBooster(agg_message)

        agg_message = self.lr(t.cat([agg_message, message_atom, input_atom], 1))

        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = t.stack(mol_vecs, dim=0)

        return mol_vecs

    def MessageBooster(self, agg_message):
        return agg_message.sum(dim=1) * agg_message.max(dim=1)[0]

    def _unpack_inputs(self, input):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = input.get_components()
        #print(f_bonds)
        #print(f_bonds.size())
        #print(self.opt.args['CUDA_VISIBLE_DEVICES'])
        f_atoms, f_bonds, a2b, b2a, b2revb = (
                f_atoms.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                f_bonds.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                a2b.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                b2a.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')),
                b2revb.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')))

        return f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds

class MPN(nn.Module):
    def __init__(self, opt):
        super(MPN, self).__init__()
        self.opt = opt
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + (not opt.args['atom_messages']) * self.atom_fdim
        self.encoder = MPNEncoder(self.opt, self.atom_fdim, self.bond_fdim)

    def forward(self, input):
        input = mol2graph(input, self.opt)
        output = self.encoder.forward(input)

        return output

######################################################################################


class CMPNNModel(nn.Module):
    # A CMPNN Model includes a message passing network following by a FCN.

    def __init__(self, classification: bool, multiclass: bool, opt):
        super(CMPNNModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

        self.opt = opt
        self.hidden_size = opt.args['FPSize']
        self.num_classes = opt.args['ClassNum']
        self.dataset_type = opt.args['dataset_type']
        self.output_size = opt.args['OutputSize']
        self.ffn_hidden_size = opt.args['ffn_hidden_size']

        self.only_extract_feature = opt.args['only_extract_feature']

        if self.dataset_type == 'multicalss':
            self.multiclass == True

        self.create_encoder()
        self.create_ffn()

    def create_encoder(self):
        self.encoder = MPN(self.opt)

    def create_ffn(self):
        first_linear_dim = self.hidden_size * 1

        dropout = nn.Dropout(self.opt.args['DropRate'])
        activation = get_activation_function(self.opt)

        # Create FNN Layers
        if self.opt.args['ffn_num_layers'] == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.ffn_hidden_size)
            ]
            for _ in range(self.opt.args['ffn_num_layers'] - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(self.ffn_hidden_size, self.output_size)
            ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self, input):
        # An encoder to extract information of a graph
        # and a FCN as task layer to make prediction
        # output = self.ffn(self.encoder(input))
        # print(input)
        output = self.encoder(input)
        if self.only_extract_feature:
            # print(f"size of output is: {output.size()}")
            return output

        # self.ffn has the same function with the DNN classifier model
        # If we only need to extract features, ffn is not needed.
        output = self.ffn(output)
        # output layer
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0),-1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output

