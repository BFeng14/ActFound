import numbers
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
import torch


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict


class MetaLinearLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_dim: The shape of the input data
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(hidden_dim, input_dim))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(task_feat.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.num_updates, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.num_updates, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.num_updates, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.num_updates, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum
        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.cuda(), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.cuda(), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying a layer norm function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params["bias"]
        else:
            bias = self.bias
            #print('no inner loop params', self)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MetaLinearNormLayerReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_dim:
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaLinearNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_dim = input_dim
        self.args = args
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros([2, self.input_dim])

        out = x

        self.linear = MetaLinearLayer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, use_bias=self.use_bias)

        out = self.linear(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)
        # out = F.relu(out)


        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        linear_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            linear_params = params['linear']

        out = x


        out = self.linear(out, params=linear_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)
        # out = F.relu(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class AssayFCNReLUNormNetworkReg(nn.Module):
    def __init__(self, input_shape, args, meta=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param args: A named tuple containing the system's hyperparameters.
        :param meta: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(AssayFCNReLUNormNetworkReg, self).__init__()
        self.total_layers = 0
        self.args = args
        self.hidden_dim = args.hid_dim
        self.num_stages = args.num_stages

        self.input_shape = list(input_shape)

        self.meta = meta

        self.build_network()
        self.sigmoid = nn.Sigmoid()

        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        out1 = torch.zeros(self.input_shape)
        out2 = torch.zeros(self.input_shape)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            self.layer_dict['linear{}'.format(i)] = MetaLinearNormLayerReLU(input_dim=out1.shape[-1],
                                                                            hidden_dim=self.hidden_dim,
                                                                            use_bias=True,
                                                                            args=self.args,
                                                                            normalization=True,
                                                                            meta_layer=self.meta,
                                                                            no_bn_learnable_params=False)

            out1 = self.layer_dict['linear{}'.format(i)](out1, training=True, num_step=0)
            out2 = self.layer_dict['linear{}'.format(i)](out2, training=True, num_step=0)

        # out = torch.cat([out1, out2], dim=-1)
        out = out1 - out2
        # self.layer_dict['linear'] = MetaMaskLayer(input_dim=self.hidden_dim)
        self.layer_dict['linear'] = MetaLinearLayer(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, use_bias=True)
        out = self.layer_dict['linear'](out)

        # self.layer_dict['linearrec'] = MetaLinearLayer(input_dim=self.hidden_dim, hidden_dim=out1.shape[-1], use_bias=True)
        print("FCNNetwork build", out.shape)

    def forward(self, x, split=None, num_step=0, params=None, training=False, backup_running_statistics=False, mixup=None, lam=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param x: pairs idex
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        input_dim = self.input_shape[-1]
        out = x
        nx = int(out.shape[0] / 2)

        for i in range(self.num_stages):
            out = self.layer_dict['linear{}'.format(i)](out, params=param_dict['linear{}'.format(i)],
                                                         training=training,
                                                         backup_running_statistics=backup_running_statistics,
                                                         num_step=num_step)

        out = self.layer_dict['linear'](out, param_dict['linear'])
        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['linear{}'.format(i)].restore_backup_stats()


class FCNReLUNormNetworkQSAR(nn.Module):
    def __init__(self, input_shape, args, meta=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param args: A named tuple containing the system's hyperparameters.
        :param meta: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(FCNReLUNormNetworkQSAR, self).__init__()
        self.total_layers = 0
        self.args = args
        self.hidden_dim = args.hid_dim
        self.num_stages = args.num_stages

        self.input_shape = list(input_shape)

        self.meta = meta

        self.build_network()

        # print("meta network params")
        # for name, param in self.named_parameters():
        #     print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        out = torch.zeros(self.input_shape)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            self.layer_dict['linear{}'.format(i)] = MetaLinearNormLayerReLU(input_dim=out.shape[-1],
                                                                            hidden_dim=self.hidden_dim,
                                                                            use_bias=True,
                                                                            args=self.args,
                                                                            normalization=True,
                                                                            meta_layer=self.meta,
                                                                            no_bn_learnable_params=False)

            out = self.layer_dict['linear{}'.format(i)](out, training=True, num_step=0)

        self.layer_dict['linear'] = MetaLinearLayer(input_dim=self.hidden_dim, hidden_dim=1, use_bias=False)
        out = self.layer_dict['linear'](out)
        if self.args.input_celline:
            self.layer_dict['linear_cell_1'] = MetaLinearLayer(input_dim=199, hidden_dim=self.hidden_dim, use_bias=False)
            # self.layer_dict['linear_cell_2'] = MetaLinearLayer(input_dim=256, hidden_dim=self.hidden_dim, use_bias=False)
        print("FCNNetwork build", out.shape)

    def forward(self, x, num_step, mixup=False, lam=None, params=None, training=False, backup_running_statistics=False, cellparam=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        nx = int(out.shape[0] / 2)
        if mixup:
            layer_mix = self.args.rng.randint(0, self.num_stages + 1)

        if mixup and layer_mix == 0:
            out = lam * out[:nx, :] + (1 - lam) * out[nx:, :]

        for i in range(self.num_stages):
            out = self.layer_dict['linear{}'.format(i)](out, params=param_dict['linear{}'.format(i)],
                                                         training=training,
                                                         backup_running_statistics=backup_running_statistics,
                                                         num_step=num_step)
            if mixup and layer_mix == (i + 1):
                out = lam * out[:nx, :] + (1 - lam) * out[nx:, :]

        if self.args.input_celline:
            cell_embed = self.layer_dict['linear_cell_1'](cellparam)
            cell_embed = F.relu(cell_embed)
            cell_embed = self.layer_dict['linear_cell_2'](cell_embed)
            cell_embed = F.sigmoid(cell_embed)
            out = out * cell_embed
        out_value = self.layer_dict['linear'](out, param_dict['linear'])[:, 0]
        return out, out_value

    def forward_pred(self, out, assay_cls_idx=None, params=None):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out_value = self.layer_dict['linear'](out, param_dict['linear'])[:, 0]
        return out_value

    def forward_feat(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        for i in range(self.num_stages):
            out = self.layer_dict['linear{}'.format(i)](out, params=param_dict['linear{}'.format(i)],
                                                         training=training,
                                                         backup_running_statistics=backup_running_statistics,
                                                         num_step=num_step)
        return out*self.layer_dict['linear'].weights, self.layer_dict['linear'](out, param_dict['linear'])[:, 0]

    def load_cluster_param(self, cluster_param):
        self.layer_dict['linear_cluster'].weights = nn.Parameter(cluster_param)

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['linear{}'.format(i)].restore_backup_stats()

