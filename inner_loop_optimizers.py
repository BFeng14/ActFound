import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class GradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, device, learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(GradientDescentLearningRule, self).__init__()
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = torch.ones(1) * learning_rate
        self.learning_rate.cuda()

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, tau=0.9):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        for key in names_weights_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - self.learning_rate * \
                                              names_grads_wrt_params_dict[
                                                  key]

        return updated_names_weights_dict


class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, args, total_num_inner_loop_steps, use_learnable_learning_rates, init_learning_rate=1e-3):
        """Creates a new learning rule object.
        Args:
            init_learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
        """
        super(LSLRGradientDescentLearningRule, self).__init__()
        print(init_learning_rate)
        assert init_learning_rate > 0., 'learning_rate should be positive.'

        self.args = args
        self.init_learning_rate = torch.ones(1) * init_learning_rate
        self.init_learning_rate.cuda()
        self.total_num_inner_loop_steps = total_num_inner_loop_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates
        self.kname = "layer_dict.linear.weights".replace(".", "-")

    def initialise(self, names_weights_dict):
        self.names_learning_rates_dict = nn.ParameterDict()
        self.names_learning_rates_dict[self.kname] = nn.Parameter(
            data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate,
            requires_grad=self.use_learnable_learning_rates)
        # self.names_learning_rates_k = nn.ParameterDict()
        # self.names_learning_rates_b = nn.ParameterDict()
        # for idx, (key, param) in enumerate(names_weights_dict.items()):
        #     self.names_learning_rates_k[key.replace(".", "-")] = nn.Parameter(
        #         data=torch.ones(self.total_num_inner_loop_steps + 1) * 0.0003,
        #         requires_grad=self.use_learnable_learning_rates)
        #     self.names_learning_rates_b[key.replace(".", "-")] = nn.Parameter(
        #         data=torch.ones(self.total_num_inner_loop_steps + 1) * (self.init_learning_rate - 0.0003),
        #         requires_grad=self.use_learnable_learning_rates)


    def reset(self):

        # for key, param in self.names_learning_rates_dict.items():
        #     param.fill_(self.init_learning_rate)
        pass

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step, sup_number=None):
        """Applies a single gradient descent update to all parameters.
        All parameter updates are performed using in-place operations and so
        nothing is returned.
        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        updated_names_weights_dict = dict()
        # sup_number = torch.tanh(torch.sqrt(torch.tensor(sup_number).float()/100.)).cuda()
        # sup_number = torch.sqrt(torch.tensor(sup_number).float()).cuda()
        # for key in names_grads_wrt_params_dict.keys():
        #     lr_k = self.names_learning_rates_k[key.replace(".", "-")][num_step]
        #     lr_b = self.names_learning_rates_b[key.replace(".", "-")][num_step]
        #     updated_names_weights_dict[key] = names_weights_dict[key] - \
        #                                       (lr_k*sup_number+lr_b) * names_grads_wrt_params_dict[key]
        if "transfer" in self.args.model_name.lower():
            lr = self.args.transfer_lr
        else:
            if self.args.use_byhand_lr:
                lr = [0.012, 0.010, 0.008, 0.004, 0.002, 0.001][num_step]
            else:
                lr = self.names_learning_rates_dict[self.kname][num_step]
        for key in names_grads_wrt_params_dict.keys():
            updated_names_weights_dict[key] = names_weights_dict[key] - \
                                              lr * names_grads_wrt_params_dict[key]# self.names_learning_rates_dict[self.kname][num_step]

        return updated_names_weights_dict

