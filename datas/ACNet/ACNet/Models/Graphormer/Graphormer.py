import torch
import math
import torch.nn as nn
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """
    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers,flag
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.flag = flag
        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        #initate parameters
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        '''
        batch_data:
            attn_bias: [n_graph, n_node+1, n_node+1], the shortest path beyond the spatial_pos_max: -inf; else:0
            spatial_pos: [n_graph, n_node, n_node],the shortest path between nodes in the graph
            x: [n_graph, n_node, n_node_features], node feature
            in_degree: [n_graph, n_node]
            out_degree: [n_graph, n_node]
            edge_input: [n_graph, n_node, n_node, multi_hop_max_dist, n_edge_features]
            attn_edge_type: [n_graph,n_node,n_node,n_edge_features], edge feature
        '''
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feature + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        if self.flag and perturb is not None:
            node_feature += perturb
        # eq5, node feature + centrality encoding #
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature

class GraphAttnBias(nn.Module):
    '''
    Compute attention bias for each head.
    '''
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.num_edges = num_edges+1
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # initiate parameters
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        # attn_bias: an all zero tensor initially with shape: [node_num+1, node_num+1], for A_{ij}
        # +1 is for the 'graph' node.
        # spatial_pos: a tensor with shape [node_num, node_num], for Phi(vi,vj)
        # note: The first dim should be batch, so their shapes are [batch, x, x].


        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )
        # edge_input shape: [batch, node_num, node_num, max_dist, edge_feat]
        # attn_edge_type: [batch, node_num, node_num, edge_feat]
        # edge_input stores the edge_attrs of all of the edges in the SP to travel from i to j.
        # edge_attrs are converted. (i.e. added offsets)
        # attn_edge_type stores the edge_attrs of all of the edges in the graph.
        # attn_edge_type are converted.

        # print(f"edge_input value: {edge_input[0,0,2,:,:]}")
        # print(f"attn_edge_type value: {attn_edge_type[0,0,1,:]}")
        # print(f"attn_edge_type value: {attn_edge_type[0,1,2,:]}")
        # raise RuntimeError

        n_graph, n_node = x.size()[:2]

        #graph_attn_bias
        #add virtual node [VNode] as the whole representation of graph feature
        graph_attn_bias = attn_bias.clone()      # [batch, node_num+1, node_num+1]
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [batch,1,nodenum+1,nodenum+1] to [n_graph, n_head, n_node+1, n_node+1], all zeros.

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        # print(f"spatial_pos size: {spatial_pos.size()}")
        # print(f"spatial_pos max: {spatial_pos.max()}")
        # print(f"spatial_pos min: {spatial_pos.min()}")
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # spatial_pos is a tensor with [batch, n_node, n_node], each element indicates the SPD between i,j
        # spatial_pos_encoder is an nn.Embedding to embed the SPD value to a tensor with n_head length.
        # the output of the encoder should be [n_graph, n_node, n_node, n_head]
        # then, by permute, it transfers to [n_graph, n_head, n_node, n_node]

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        # [n_graph, n_head, n_node+1, n_node+1], so for 1:, it's n_node
        # what about the Vnode?
        # Obviously, node index 0 is the Vnode in graph_attn_bias.

        # reset spatial pos here
        # [Vnode] is connected to all other nodes in graph, the distance of shortest path is 1 for ([Vnode],v_j)
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # the SPD of (Vnode, v_j) is 1, so that the embedding of such SPD is the weight of the embedding module.
        # the SPD of Vnode and all of the nodes in the graph (including Vnode it self) is set to be 1
        # And the bias is calculated by a linear embedding layer.
        # Remember, Vnode is node 0.
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop":
            # edge_type set to be multi_hop means that it considers relationships between multi-hop neighbor nodes.
            spatial_pos_ = spatial_pos.clone()
            # spatial_pos_: [n_graph, n_node, n_node]
            # after batch, the pads are 0, and the exact 0 hops (cannot reach or the node itself) are 1.

            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            # now, the hops are their exact hops.
            # and the pads are 1, for the node pairs that are unreached.
            # Remember, no Vnode added here.
            # print(f"edge_input size before clamp: {edge_input.size()}")

            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                # set the values out of range [min, max] to be min or max.
                # why?
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]

            # [n_graph, n_node, n_node, max_dist, edge_feat] -> [n_graph, n_node, n_node, max_dist, edge_feat, n_head]
            # -> [n_graph, n_node, n_node, max_dist, n_head]
            # averaged by edge_feat numbers.
            # print(f"edge_input size: {edge_input.size()}")
            # print(f"edge_encoder: {self.edge_encoder}")
            # print(f"device of edge_input: {edge_input.device}")
            # print(f"max of edge_input: {edge_input.max()}")
            # print(f"min of edge_input: {edge_input.min()}")
            # raise RuntimeError
            edge_input = self.edge_encoder(edge_input).mean(-2)     #encoding
            # raise RuntimeError
            max_dist = edge_input.size(-2)
            # [n_graph, n_node, n_node, max_dist, n_head] -> [max_dist, n_graph, n_node, n_node, n_head]
            # -> [max_dist, -1, n_head], flatten all of the SP edge embeddings.
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            # torch.bmm: (b*n*m) bmm (b*m*p) = (b*n*p), batched mat multiply
            # here:
            # edge_input_flag: [max_dist, n_graph*n_node*n_node, n_head]
            # edge_dis_encoder.weight: [num_edge_dis*n_head*n_head, 1]
            # weight.reshape[]: [max_dist, n_head, n_head]
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            # edge_input_flag: [max_dist, n_graph*n_node*n_node, n_head]
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            # edge_input: [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
            # sum the max_dist embeddings, and devide the number of edges in SP, i.e. N in eq.7
            # edge_input: [n_graph, n_head, n_node, n_node]
        else:
            # if not multi-hop, it indicates that only edges of directly linked nodes are considered.
            # According to eq.7, the features of all edges in a SP will be computed into a scalar.
            # so, after the edge_encoder, the shape is:
            # [n_graph, n_node, n_node, edge_feat] -> [n_graph, n_node, n_node, edge_feat, n_head]
            # by .mean(-2), it averages all embeddings of values of an edge: [n_graph, n_node, n_node, n_head]
            # and by permute: [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset   # reset? why?

        return graph_attn_bias

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        '''
        hidden_size: output of hidden layer
        attention_dropout_rate: dropout rate inside attention for training
        num_heads: number of heads to repeat the same attention structure
        '''
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        # split into several heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        # 1/sqrt(d_k)
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        '''
        add attn_bias, then softmax and matmul
        '''
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # eq7,softmax and matmul after adding bias
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        #different from original transformer, LayerNorm before self-attention and FFN
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class Graphormer(nn.Module):
    def __init__(
            self,
            num_encoder_layers,
            num_attention_heads,
            embedding_dim,
            dropout_rate,
            intput_dropout_rate,
            ffn_dim,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            opt,
            flag=False,
            mode='Pred'
            ):
        super(Graphormer, self).__init__()
        self.num_heads = num_attention_heads
        self.opt=opt
        self.mode = mode
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=opt.args['num_atoms'],
            num_in_degree=opt.args['num_in_degree'],
            num_out_degree=opt.args['num_out_degree'],
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            flag=flag
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=opt.args['num_atoms'],
            num_edges=opt.args['num_edges'],
            num_spatial=opt.args['num_spatial'],
            num_edge_dis=opt.args['num_edge_dis'],
            hidden_dim=embedding_dim,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
        )
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(embedding_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_attention_heads)
                    for _ in range(num_encoder_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(embedding_dim)
        # if dataset_name == 'PCQM4M-LSC':
        #     self.out_proj = nn.Linear(embedding_dim, 1)
        # else:
            # self.downstream_out_proj = nn.Linear(
            #     embedding_dim, get_dataset(dataset_name)['num_class'])
        if self.mode == 'Pred':
            self.downstream_out_proj = nn.Linear(
                embedding_dim, opt.args['OutputSize'])

        self.flag = flag
        self.apply(lambda module: init_params(module, n_layers=num_encoder_layers))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batched_data,perturb = None):
        # raise RuntimeError
        x = self.graph_node_feature(batched_data,perturb)
        attn_bias = self.graph_attn_bias(batched_data)
        # Calculating #
        # transformer encoder eq 8,9
        output = self.input_dropout(x)
        for enc_layer in self.layers:
            output = enc_layer(output, attn_bias)
        output = self.final_ln(output) # h(l)
        # output part
        # The whole graph representation is the feature of Vnode at the last layer.
        # if self.dataset_name == 'PCQM4M-LSC':
        #     # get whole graph rep
        #     output = self.out_proj(output[:, 0, :])
        # else:
        #     output = self.downstream_out_proj(output[:, 0, :])
        if self.mode == 'Pred':
            output = self.downstream_out_proj(output[:, 0, :])
            if self.opt.args['ClassNum'] != 1:
                if not self.training:
                    output = self.softmax(output)
        else:
            output = output[:,0,:]
        return output

# if __name__ == '__main__':
#     model = Graphormer(
#             12,
#             32,
#             512,
#             0.1,
#             0.1,
#             512,
#             'ogbg-molpcba',
#             'multi_hop',
#             10,
#             0.1,
#             flag=False)
#     print(model)
