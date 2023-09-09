import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg, AssayAttn
from inner_loop_optimizers import LSLRGradientDescentLearningRule


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng



import torch.distributed as dist
from torch.optim.optimizer import Optimizer

class DistributedOptimizer(Optimizer):

    """

    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/optim.py)

    **Description**

    Synchronizes the gradients of a model across replicas.

    At every step, `Distributed` averages the gradient across all replicas
    before calling the wrapped optimizer.
    The `sync` parameters determines how frequently the parameters are
    synchronized between replicas, to minimize numerical divergences.
    This is done by calling the `sync_parameters()` method.
    If `sync is None`, this never happens except upon initialization of the
    class.

    **Arguments**

    * **params** (iterable) - Iterable of parameters.
    * **opt** (Optimizer) - The optimizer to wrap and synchronize.
    * **sync** (int, *optional*, default=None) - Parameter
      synchronization frequency.

    **References**

    1. Zinkevich et al. 2010. “Parallelized Stochastic Gradient Descent.”

    **Example**

    ~~~python
    opt = optim.Adam(model.parameters())
    opt = Distributed(model.parameters(), opt, sync=1)

    opt.step()
    opt.sync_parameters()
    ~~~

    """

    def __init__(self, params, opt, sync=None):
        self.world_size = dist.get_world_size()
        print("world_size", self.world_size)
        self.rank = dist.get_rank()
        self.opt = opt
        self.sync = sync
        self.iter = 0
        defaults = {}
        super(DistributedOptimizer, self).__init__(params, defaults)
        self.sync_parameters()

    def sync_parameters(self, root=0):
        """
        **Description**

        Broadcasts all parameters of root to all other replicas.

        **Arguments**

        * **root** (int, *optional*, default=0) - Rank of root replica.

        """
        if self.world_size > 1:
            for group in self.param_groups:
                for p in group['params']:
                    dist.broadcast(p.data, src=root)

    def step(self):
        if self.world_size > 1:
            num_replicas = float(self.world_size)
            # Average all gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad

                    # Perform the averaging
                    dist.all_reduce(d_p)
                    d_p.data.mul_(1.0 / num_replicas)

        # Perform optimization step
        self.opt.step()
        self.iter += 1

        if self.sync is not None and self.iter >= self.sync:
            self.sync_parameters()
            self.iter = 0


class MAMLRegressor(nn.Module):
    def __init__(self, input_shape, args):

        super(MAMLRegressor, self).__init__()
        self.args = args
        self.batch_size = args.meta_batch_size
        self.current_epoch = 0
        self.input_shape = input_shape

        self.mixup = self.args.mixup

        self.rng = set_torch_seed(seed=0)

        self.args.rng = self.rng

        if self.args.assay_desc:
            self.assay_attn = AssayAttn(in_dim=args.assay_feat_dim, out_dim=args.hid_dim * 2)

        if self.args.new_ddg or self.args.qsar:
            self.regressor = FCNReLUNormNetworkQSAR(input_shape=self.input_shape, args=self.args, meta=True).cuda()
        else:
            self.regressor = AssayFCNReLUNormNetworkReg(input_shape=self.input_shape, args=self.args, meta=True).cuda()

        self.task_learning_rate = args.update_lr

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(args=args, init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.num_updates,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.regressor.named_parameters()))
        self.temp = 0.2 #nn.Parameter(data=torch.tensor(0.1))
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape, value.requires_grad)

        self.args = args
        self.cuda()
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)

        self.all_task_feat = None
        opt = optim.Adam(self.trainable_parameters(), lr=args.meta_lr, amsgrad=False)
        if args.ddp:
            self.optimizer = DistributedOptimizer(self.trainable_parameters(), opt, sync=1)
        else:
            self.optimizer = opt
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,
                                                              T_max=self.args.metatrain_iterations,
                                                              eta_min=self.args.min_learning_rate)
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        if self.args.input_celline:
            self.cellline_feats = pickle.load(open(self.args.cell_line_feat, "rb"))

        if self.args.cluster_meta:
            if self.args.cluster_load:
                # self.optimizer = optim.Adam(self.get_weighted_training_param(), lr=args.meta_lr, amsgrad=False)
                self.assay_idx_2_cls = pickle.load(open(args.assay_idx_2_cls, "rb"))
                state_dict = torch.load(args.cls_model_file)
                self.load_state_dict(state_dict, strict=False)
                first_step_center = state_dict['regressor.layer_dict.linear.weights']
                cluster_center = np.load(args.cluster_center) + np.random.rand(10, 2048) * 0.003
                cluster_center = np.concatenate([np.zeros([1, cluster_center.shape[1]]), cluster_center], axis=0)
                cluster_center = torch.tensor(cluster_center).float().cuda() + first_step_center
                # cluster_center = first_step_center.repeat(args.num_clusters + 1, 1)
                self.regressor.load_cluster_param(cluster_center)
            else:
                self.assay_idx_2_cls = pickle.load(open(args.assay_idx_2_cls, "rb"))


    def cossim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def load_all_task_feat(self, all_task_feat):
        self.all_task_feat = torch.Tensor(all_task_feat).float().cuda()

    def get_per_step_loss_importance_vector(self):

        loss_weights = np.ones(shape=(self.args.num_updates)) * (
                1.0 / self.args.num_updates)
        decay_rate = 1.0 / self.args.num_updates / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.num_updates
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.num_updates - 1) * decay_rate),
            1.0 - ((self.args.num_updates - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).cuda()
        return loss_weights

    def get_weighted_training_param(self):
        params = self.regressor.named_parameters()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    yield param
                else:
                    if self.args.cluster_meta:
                        if "layer_dict.linear_cluster.bias" in name or "layer_dict.linear_cluster.weights" in name:
                            yield param
                    else:
                        if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                            yield param

    def get_inner_loop_parameter_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.cuda()
                else:
                    if self.args.cluster_meta:
                        if "layer_dict.linear_cluster.bias" in name or "layer_dict.linear_cluster.weights" in name:
                            param_dict[name] = param.cuda()

                    else:
                        if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                            param_dict[name] = param.cuda()

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx, sup_number):

        self.regressor.zero_grad(names_weights_copy)

        # if loss >= 10.:
        #     return names_weights_copy

        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                        create_graph=use_second_order)
            # clip_value = 0.1
            # for grad in grads:
            #     grad.data.clamp_(min=-clip_value, max=clip_value)
            names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))
            names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                         names_grads_wrt_params_dict=names_grads_wrt_params,
                                                                         num_step=current_step_idx,
                                                                         sup_number=sup_number)
        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, loss_weights=None):
        losses = dict()
        total_losses = torch.stack(total_losses)
        if loss_weights is None:
            loss_weights = torch.ones_like(total_losses).cuda()
        else:
            loss_weights = torch.FloatTensor(loss_weights).cuda()
        losses['loss'] = torch.mean(total_losses * loss_weights)
        return losses

    def get_init_weight(self):
        init_weight = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
        if self.args.cluster_meta:
            init_weight = init_weight["layer_dict.linear_cluster.weights"].detach().cpu().numpy()
        else:
            init_weight = init_weight["layer_dict.linear.weights"].detach().cpu().numpy().squeeze()
        return init_weight

    def clip_fun(self, ddg):
        return torch.tanh(ddg / 3) * 3
    
    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase,
                same_sup_tgt=False):

        xs, ys, splits, assay_idxes, assay_weight, _ = data_batch

        total_losses = []
        per_task_target_preds = [[] for i in range(len(xs))]
        self.regressor.zero_grad()
        final_weights = []
        sup_losses = []
        uncertainty_all = []
        start = 0

        for task_id, (x_task, y_task, split, assay_idx) in \
                enumerate(zip(xs,
                              ys,
                              splits,
                              assay_idxes
                              )):

            # # first of all, put all tensors to the device
            y_train = y_task[torch.nonzero(split)]

            if self.args.datasource == "fsmol":
                std_y_train = y_train.std()
            else:
                std_y_train = max(0.2, y_train.std())
            mean_y_train = y_train.mean()

            if self.args.qsar:
                if self.args.datasource == "bdb":
                    bias_y = 6.75021
                elif self.args.datasource == "drug":
                    bias_y = -2.2329
            else:
                bias_y = 0.

            # bias_y = 0
            # std_y_train = 1.
            # mean_y_train = 0.

            y_task = (y_task.float().cuda() - mean_y_train) / std_y_train + bias_y
            x_task = x_task.float().cuda()
            self.clear_cache()
            self.training_phase = training_phase
            self.assay_cls_idx_cache = None

            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            support_loss_each_step = []
            support_loss_first_step = None
            if not (self.args.transfer_l and training_phase) and (self.args.qsar or self.args.new_ddg) and (not self.args.input_celline):
                sup_num = torch.sum(split)
                names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
                for num_step in range(num_steps):
                    # if num_step >= num_steps-1:
                    #     num_step = num_steps-1
                    support_loss, support_preds, _ = self.net_forward(x=x_task,
                                                                      y=y_task,
                                                                      assay_idx=assay_idx,
                                                                      split=split,
                                                                      is_support=True,
                                                                      same_sup_tgt=same_sup_tgt,
                                                                      dropout=False,
                                                                      weights=names_weights_copy,
                                                                      backup_running_statistics=
                                                                      True if (num_step == 0) else False,
                                                                      training=True, num_step=num_step)
                    support_loss_each_step.append(support_loss.detach().cpu().item())
                    if support_loss_first_step is None:
                        support_loss_first_step = support_loss.detach().cpu().item()
                    support_loss_last_step = support_loss.detach()
                    if support_loss >= support_loss_first_step * 10 or sup_num <= 5:
                        # print("assay {} exploded, skip inner loop update".format(assay_idx))
                        pass
                    else:
                        try:
                            names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                              names_weights_copy=names_weights_copy,
                                                                              use_second_order=use_second_order,
                                                                              current_step_idx=num_step,
                                                                              sup_number=torch.sum(split))
                        except Exception as e:
                            print(e)
                            pass

                    if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                        target_loss, target_preds, _ = self.net_forward(x=x_task,
                                                                        y=y_task,
                                                                        assay_idx=assay_idx,
                                                                        split=split,
                                                                        same_sup_tgt=same_sup_tgt,
                                                                        dropout=True,
                                                                        weights=names_weights_copy,
                                                                        backup_running_statistics=False, training=True,
                                                                        num_step=num_step, mixup=self.mixup)

                        task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                    else:
                        if num_step == (self.args.num_updates - 1) and training_phase:
                            target_loss, target_preds, _ = self.net_forward(x=x_task,
                                                                            y=y_task,
                                                                            assay_idx=assay_idx,
                                                                            split=split,
                                                                            same_sup_tgt=same_sup_tgt,
                                                                            dropout=True,
                                                                            weights=names_weights_copy,
                                                                            backup_running_statistics=False,
                                                                            training=True,
                                                                            num_step=num_step, mixup=self.mixup)
                            task_losses.append(target_loss)
            else:
                names_weights_copy = None
            # not used for compute loss fun
            target_loss, target_preds, uncertainty = self.net_forward(x=x_task,
                                                                      y=y_task,
                                                                      assay_idx=assay_idx,
                                                                      split=split,
                                                                      same_sup_tgt=same_sup_tgt,
                                                                      dropout=False,
                                                                      weights=names_weights_copy,
                                                                      backup_running_statistics=False, training=True,
                                                                      num_step=num_steps - 1)

            if self.args.transfer_l or self.args.input_celline:
                task_losses.append(target_loss)
            elif not training_phase:
                task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # uncertainty_all.append(uncertainty.detach().squeeze().cpu().numpy())
            task_losses = torch.sum(torch.stack(task_losses))  # + rec_loss
            total_losses.append(task_losses)
            final_weights.append(names_weights_copy)
            sup_losses.append({"support_loss_each_step": support_loss_each_step})

            if not training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)

        return losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all

    def forward_knn(self, data_batch, data_batch_knn, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase,
                same_sup_tgt=False):

        xs, ys, splits, assay_idxes, assay_weight = data_batch
        xs_knn, ys_knn, splits_knn, assay_idxes_knn, assay_weight_knn = data_batch_knn

        total_losses = []
        per_task_target_preds = [[] for i in range(len(xs))]
        self.regressor.zero_grad()
        final_weights = []
        sup_losses = []
        uncertainty_all = []
        start = 0

        for task_id, (x_task, y_task, split, assay_idx) in \
                enumerate(zip(xs,
                              ys,
                              splits,
                              assay_idxes
                              )):

            # first of all, put all tensors to the device
            y_train = y_task[torch.nonzero(split)]
            if not self.args.qsar:
                if self.args.datasource == "fsmol":
                    std_y_train = y_train.std()
                else:
                    std_y_train = max(0.2, y_train.std())
                mean_y_train = y_train.mean()
            else:
                std_y_train = 1.
                mean_y_train = 0.

            y_task = (y_task.float().cuda() - mean_y_train) / std_y_train
            x_task = x_task.float().cuda()
            self.clear_cache()
            self.training_phase = training_phase
            self.assay_cls_idx_cache = None

            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            support_loss_first_step = None
            if not (self.args.transfer_l and training_phase) and (self.args.qsar or self.args.new_ddg) and (
            not self.args.input_celline):
                names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
                for num_step in range(num_steps):
                    # if num_step >= num_steps-1:
                    #     num_step = num_steps-1
                    support_loss_1, support_preds, _ = self.net_forward(x=x_task,
                                                                      y=y_task,
                                                                      assay_idx=assay_idx,
                                                                      split=split,
                                                                      is_support=True,
                                                                      same_sup_tgt=same_sup_tgt,
                                                                      dropout=False,
                                                                      weights=names_weights_copy,
                                                                      backup_running_statistics=
                                                                      True if (num_step == 0) else False,
                                                                      training=True, num_step=num_step)
                    support_loss_knns = []
                    for x_task_knn, y_task_knn, split_knn, weight_knn in zip(xs_knn, ys_knn, splits_knn, assay_weight_knn):
                        std_y_knn = max(0.2, y_task_knn.std())
                        mean_y_knn = y_task_knn.mean()
                        y_task_knn = (y_task_knn.float().cuda() - mean_y_knn) / std_y_knn
                        x_task_knn = x_task_knn.float().cuda()
                        split_knn = (split_knn + 1.) / (split_knn + 1.)
                        support_loss_knn, _, _ = self.net_forward(x=x_task_knn,
                                                                          y=y_task_knn,
                                                                          assay_idx=None,
                                                                          split=split_knn.long(),
                                                                          is_support=True,
                                                                          same_sup_tgt=same_sup_tgt,
                                                                          dropout=False,
                                                                          weights=names_weights_copy,
                                                                          backup_running_statistics=False,
                                                                          training=True, num_step=num_step)
                        support_loss_knns.append(support_loss_knn*weight_knn)
                    support_loss = support_loss_1 + sum(support_loss_knns)

                    if support_loss_first_step is None:
                        support_loss_first_step = support_loss.detach().cpu().item()
                    support_loss_last_step = support_loss.detach()
                    if support_loss >= support_loss_first_step * 10:
                        # print("assay {} exploded, skip inner loop update".format(assay_idx))
                        pass
                    else:
                        try:
                            names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                              names_weights_copy=names_weights_copy,
                                                                              use_second_order=use_second_order,
                                                                              current_step_idx=num_step,
                                                                              sup_number=torch.sum(split))
                        except Exception as e:
                            print(e)
                            pass

            else:
                names_weights_copy = None
            # not used for compute loss fun
            target_loss, target_preds, uncertainty = self.net_forward(x=x_task,
                                                                      y=y_task,
                                                                      assay_idx=assay_idx,
                                                                      split=split,
                                                                      same_sup_tgt=same_sup_tgt,
                                                                      dropout=False,
                                                                      weights=names_weights_copy,
                                                                      backup_running_statistics=False, training=True,
                                                                      num_step=num_steps - 1)

            if self.args.transfer_l or self.args.input_celline:
                task_losses.append(target_loss)
            elif not training_phase:
                task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # uncertainty_all.append(uncertainty.detach().squeeze().cpu().numpy())
            task_losses = torch.sum(torch.stack(task_losses))  # + rec_loss
            total_losses.append(task_losses)
            final_weights.append(names_weights_copy)
            sup_losses.append({"support_loss_first_step": support_loss_first_step})

            if not training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)

        return losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all

    def compute_kl_divergency(self, p_mean, p_log_std):
        KL_div_0 = torch.sum(input=torch.exp(input=p_log_std))
        # x = torch.exp(input=p_log_std)
        # x_mean = torch.mean(torch.abs(x/p_mean))
        KL_div_1 = torch.sum(input=torch.square(input=p_mean))
        KL_div_2 = torch.sum(input=p_log_std)
        return (KL_div_1 + KL_div_0 - KL_div_2 - p_log_std.shape[-1]) / 2

    def self_supervise_forward(self, x, backup_running_statistics, num_step):
        rec = self.regressor.forward_rec(x, backup_running_statistics, num_step)
        rec_loss = self.bce_loss(rec, x)
        # print(rec_loss)
        return rec_loss

    def get_assay_cls_index(self, assay_idx):
        if random.random() <= 0.2:
            return 0
        else:
            return self.assay_idx_2_cls[assay_idx] + 1

    def net_forward_qsar(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                        is_support=False, mixup=None, lam=None, dropout=False, same_sup_tgt=False):
        _, out_value = self.regressor.forward(x=x, params=weights,
                                           training=training,
                                           backup_running_statistics=backup_running_statistics,
                                           num_step=num_step)
        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        support_value = out_value[sup_idx]
        tgt_value = out_value[tgt_idx]
        sup_y = y[sup_idx]
        tgt_y = y[tgt_idx]

        if is_support or same_sup_tgt:
            preds = support_value
            loss = torch.mean((preds - sup_y) ** 2)
        else:
            preds = tgt_value
            loss = torch.mean((preds - tgt_y) ** 2)

        loss = torch.sqrt(loss)
        return loss, preds, torch.abs(preds)

    def net_forward_cell(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                        is_support=False, mixup=None, lam=None, dropout=False, same_sup_tgt=False):

        cellparam = torch.tensor(self.cellline_feats[assay_idx]).float().cuda()
        _, out_value = self.regressor.forward(x=x, params=weights,
                                           training=training,
                                           backup_running_statistics=backup_running_statistics,
                                           num_step=num_step, cellparam=cellparam)
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        tgt_y = y[tgt_idx]
        tgt_value = out_value[tgt_idx]
        tgt_x = x[tgt_idx]
        ddg_pred = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
        ddg_real = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
        sup_sim_mat = self.get_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_x.shape[0]).cuda()
        _, topk_idx = torch.topk(sup_sim_mat, dim=0, k=sup_sim_mat.shape[0] // 2)

        loss = torch.sqrt(self.robust_square_error(ddg_pred, ddg_real, topk_idx))
        loss_dg = torch.sqrt(torch.mean((tgt_value - tgt_y) ** 2))
        return loss_dg, tgt_value, torch.abs(out_value)

    def get_cached_sim(self, split, vname):
        return self.sim_cache[split].get(vname)

    def save_cached_sim(self, split, vname, v):
        self.sim_cache[split][vname] = v

    def clear_cache(self):
        self.sim_cache = {"sup":{}, "tgt":{}}
    
    def get_sim_matrix(self, a, b):
        a_bool = (a > 0.).float()
        b_bool = (b > 0.).float()
        and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
        or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
        sim = and_res / or_res
        return sim

    def robust_square_error(self, a, b, topk_idx):
        abs_diff = torch.abs(a - b)
        square_mask = (abs_diff <= 2).float()
        linear_mask = 1. - square_mask
        square_error = (a - b) ** 2
        linear_error = 1. * (abs_diff - 2) + 4
        loss = square_error * square_mask + linear_error * linear_mask

        loss_select = torch.gather(loss, 0, topk_idx)
        return torch.mean(loss_select)


    def compute_ddg_sup_loss(self, x, y, split, out_value, out_embed):
        sup_idx = torch.nonzero(split)[:, 0]
        support_value = out_value[sup_idx]
        sup_x = x[sup_idx]
        sup_y = y[sup_idx]

        support_features_flat = out_embed[sup_idx]

        sup_num = torch.sum(split)

        ddg_pred = support_value.unsqueeze(-1) - support_value.unsqueeze(0)
        ddg_real = sup_y.unsqueeze(-1) - sup_y.unsqueeze(0)

        # embed_sim_matrix = sup_sim_mat / self.temp
        topk_idx = self.get_cached_sim("sup", "topk_idx")
        if topk_idx is None:
            sup_sim_mat = self.get_sim_matrix(sup_x, sup_x) - torch.eye(sup_num).cuda()
            _, topk_idx = torch.topk(sup_sim_mat, dim=0, k=sup_sim_mat.shape[0] // 2)
            self.save_cached_sim("sup", "topk_idx", topk_idx)

        embed_sim_matrix = self.cossim_matrix(support_features_flat, support_features_flat) / self.temp
        embed_sim_matrix = embed_sim_matrix - 1e9 * torch.eye(support_features_flat.shape[0]).cuda()
        embed_sim_matrix_select = torch.gather(embed_sim_matrix, 0, topk_idx)
        embed_sim_matrix_select = self.softmax(embed_sim_matrix_select)

        sup_y_repeat = sup_y.unsqueeze(-1).repeat(1, sup_num)  # [sup_num, sup_num]
        preds_all = sup_y_repeat - ddg_pred
        preds_select = torch.gather(preds_all, 0, topk_idx)
        preds = torch.sum(preds_select * embed_sim_matrix_select, dim=0)

        loss = self.robust_square_error(ddg_pred, ddg_real, topk_idx)
        loss_dg = torch.mean((preds - sup_y) ** 2)

        loss_zero = self.robust_square_error(ddg_pred * 0.0, ddg_real, topk_idx)
        loss_dg_zero = torch.mean((preds * 0.0 - sup_y) ** 2)
        if self.args.datasource == "fsmol":
            loss = loss * 0.5 + loss_dg * 0.5
        else:
            loss = loss * 0.75 + loss_dg * 0.25
        return loss


    def net_forward_ddg(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                        is_support=False, mixup=None, lam=None, dropout=False, same_sup_tgt=False):

        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        sup_x = x[sup_idx]
        tgt_x = x[tgt_idx]
        sup_y = y[sup_idx]
        tgt_y = y[tgt_idx]
        sup_num = torch.sum(split)
        tgt_num = split.shape[0] - sup_num

        if self.args.cluster_meta:
            out_embed, out_value = self.regressor.forward(x=x, params=weights,
                                                          training=training,
                                                          backup_running_statistics=backup_running_statistics,
                                                          num_step=num_step)
            if self.training_phase:
                assay_cls_idx = self.get_assay_cls_index(assay_idx)
                out_value = out_value[:, assay_cls_idx]
            else:
                if self.assay_cls_idx_cache is None:
                    assay_cls_idx = 0
                    best_r = -1000
                    best_loss = 1000
                    for i in range(self.args.num_clusters + 1):
                        # loss = self.compute_ddg_sup_loss(x, y, split, out_value[:, i], out_embed)
                        # if loss < best_loss:
                        #     assay_cls_idx = i
                        sup_idx = torch.nonzero(split)[:, 0]
                        sup_y = y[sup_idx].detach().cpu().numpy()
                        sup_pred = out_value[:, i][sup_idx].detach().cpu().numpy()
                        r = np.corrcoef(sup_y, sup_pred)[0, 1]
                        if r > best_r:
                            best_r = r
                            assay_cls_idx = i
                    self.assay_cls_idx_cache = assay_cls_idx
                    # print(assay_cls_idx)
                out_value = out_value[:, self.assay_cls_idx_cache]
            support_value = out_value[sup_idx]
            tgt_value = out_value[tgt_idx]
            support_features_flat = out_embed[sup_idx]
            query_features_flat = out_embed[tgt_idx]
        else:
            out_embed, out_value = self.regressor.forward(x=x, params=weights,
                                                          training=training,
                                                          backup_running_statistics=backup_running_statistics,
                                                          num_step=num_step)
            if self.args.mixup and not (is_support or same_sup_tgt) and self.training_phase:
                lambd = self.rng.beta(self.args.alpha, self.args.alpha)
                sup_idx_sample = sup_idx.cpu().numpy().tolist()
                random.shuffle(sup_idx_sample)
                sup_idx_sample = sup_idx_sample[:tgt_num]
                sup_idx_sample = torch.tensor(sup_idx_sample).long().cuda()
                sup_x_shuffle = x[sup_idx_sample]
                sup_y_shuffle = y[sup_idx_sample]
                mixed_set_x_task = torch.cat((sup_x_shuffle, tgt_x), dim=0)
                mixed_set_y_task = lambd * sup_y_shuffle + (1 - lambd) * tgt_y
                out_embed_mix, out_value_mix = self.regressor.forward(x=mixed_set_x_task, params=weights, mixup=True, lam=lambd,
                                                              training=training,
                                                              backup_running_statistics=backup_running_statistics,
                                                              num_step=num_step)
                support_value = out_value[sup_idx]
                support_features_flat = out_embed[sup_idx]
                tgt_x = lambd * sup_x_shuffle + (1 - lambd) * tgt_x
                tgt_y = mixed_set_y_task
                tgt_value = out_value_mix
                query_features_flat = out_embed_mix
            else:
                support_value = out_value[sup_idx]
                tgt_value = out_value[tgt_idx]
                support_features_flat = out_embed[sup_idx]
                query_features_flat = out_embed[tgt_idx]

        if is_support or same_sup_tgt:
            ddg_pred = support_value.unsqueeze(-1) - support_value.unsqueeze(0)
            ddg_real = sup_y.unsqueeze(-1) - sup_y.unsqueeze(0)

            # embed_sim_matrix = sup_sim_mat / self.temp
            # topk_idx = self.get_cached_sim("sup", "topk_idx")
            # if topk_idx is None:
            #     sup_sim_mat = self.get_sim_matrix(sup_x, sup_x) - torch.eye(sup_num).cuda()
            #     _, topk_idx = torch.topk(sup_sim_mat, dim=0, k=sup_sim_mat.shape[0] // 2)
            #     self.save_cached_sim("sup", "topk_idx", topk_idx)
            sup_sim_mat = self.get_sim_matrix(sup_x, sup_x) - torch.eye(sup_num).cuda()
            _, topk_idx = torch.topk(sup_sim_mat, dim=0, k=sup_sim_mat.shape[0] // 2)

            embed_sim_matrix = self.cossim_matrix(support_features_flat, support_features_flat) / self.temp
            embed_sim_matrix = embed_sim_matrix - 1e9 * torch.eye(support_features_flat.shape[0]).cuda()
            embed_sim_matrix_select = torch.gather(embed_sim_matrix, 0, topk_idx)
            embed_sim_matrix_select = self.softmax(embed_sim_matrix_select)

            sup_y_repeat = sup_y.unsqueeze(-1).repeat(1, sup_num)  # [sup_num, sup_num]
            preds_all = sup_y_repeat - ddg_pred
            preds_select = torch.gather(preds_all, 0, topk_idx)
            preds = torch.sum(preds_select * embed_sim_matrix_select, dim=0)

            loss = self.robust_square_error(ddg_pred, ddg_real, topk_idx)
            loss_dg = torch.mean((preds - sup_y) ** 2)

            loss_zero = self.robust_square_error(ddg_pred*0.0, ddg_real, topk_idx)
            loss_dg_zero = torch.mean((preds*0.0 - sup_y) ** 2)
            if self.args.datasource == "fsmol":
                loss = loss*0.5 + loss_dg*0.5
                loss_zero = loss_zero*0.5 + loss_dg_zero*0.5
            else:
                loss = loss_dg#loss*0.75 + loss_dg*0.25 #
                loss_zero = loss_zero * 0.75 + loss_dg_zero * 0.25 # loss_dg_zero#
        else:
            ddg_pred_1 = support_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
            ddg_real_1 = sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
            ddg_pred_2 = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
            ddg_real_2 = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)

            topk_idx = self.get_cached_sim("tgt", "topk_idx")
            if topk_idx is None:
                cross_sim_mat = self.get_sim_matrix(sup_x, tgt_x)
                _, topk_idx = torch.topk(cross_sim_mat, dim=0, k=cross_sim_mat.shape[0] // 2)
                self.save_cached_sim("tgt", "topk_idx", topk_idx)

            embed_sim_matrix = self.cossim_matrix(support_features_flat, query_features_flat) / self.temp
            sup_y_repeat = sup_y.unsqueeze(-1).repeat(1, tgt_num)  # [sup_num, tgt_num]
            preds_all = sup_y_repeat - ddg_pred_1
            
            preds_select = torch.gather(preds_all, 0, topk_idx)
            embed_sim_matrix_select = torch.gather(embed_sim_matrix, 0, topk_idx)
            embed_sim_matrix_select = self.softmax(embed_sim_matrix_select)
            preds = torch.sum(preds_select * embed_sim_matrix_select, dim=0)

            tgt_topk_idx = self.get_cached_sim("tgt", "tgt_topk_idx")
            if tgt_topk_idx is None:
                tgt_sim_mat = self.get_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_num).cuda()
                _, tgt_topk_idx = torch.topk(tgt_sim_mat, dim=0, k=tgt_sim_mat.shape[0] // 2)
                self.save_cached_sim("tgt", "tgt_topk_idx", tgt_topk_idx)

            loss_2 = self.robust_square_error(ddg_pred_2, ddg_real_2, tgt_topk_idx)
            loss_1 = self.robust_square_error(ddg_pred_1, ddg_real_1, topk_idx)
            loss_dg = torch.mean((preds - tgt_y) ** 2)
            loss_zero = 0.
            if self.args.datasource == "fsmol":
                loss = loss_2*0.15 + loss_1*0.35 + loss_dg*0.5
            else:
                loss = loss_2*0.25 + loss_1*0.5 + loss_dg*0.25

        loss = torch.sqrt(loss)
        return loss, preds, loss_zero

    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False, mixup=None, lam=None, dropout=False, same_sup_tgt=False):
        if self.args.input_celline:
            return self.net_forward_cell(x, y, split, weights, backup_running_statistics, training, num_step, assay_idx,
                                         is_support, mixup, lam, dropout, same_sup_tgt)
        elif self.args.new_ddg:
            return self.net_forward_ddg(x, y, split, weights, backup_running_statistics, training, num_step, assay_idx,
                                        is_support, mixup, lam, dropout, same_sup_tgt)
        elif self.args.qsar:
            return self.net_forward_qsar(x, y, split, weights, backup_running_statistics, training, num_step, assay_idx,
                                        is_support, mixup, lam, dropout, same_sup_tgt)

        out = self.regressor.forward(x=x, split=split, params=weights,
                                     training=training,
                                     backup_running_statistics=backup_running_statistics,
                                     num_step=num_step,
                                     mixup=mixup, lam=lam)
        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        support_features_flat = out[sup_idx]
        query_features_flat = out[tgt_idx]
        sup_y = y[sup_idx]
        tgt_y = y[tgt_idx]

        if is_support or same_sup_tgt:
            sim_matrix = self.cossim_matrix(support_features_flat, support_features_flat) / self.temp
            sim_matrix = sim_matrix - 1e9 * torch.eye(support_features_flat.shape[0]).cuda()
            sim_matrix = self.softmax(sim_matrix)
            preds = torch.matmul(sup_y, sim_matrix)
            uncertainty = torch.abs(preds)
            loss = torch.mean((preds.squeeze(-1) - sup_y) ** 2, dim=0)
            if torch.isnan(loss):
                print("loss is nan:", sim_matrix, sup_y)
        else:
            sim_matrix = self.cossim_matrix(support_features_flat, query_features_flat) / self.temp
            sim_matrix = self.softmax(sim_matrix)
            preds = torch.matmul(sup_y, sim_matrix)
            loss = torch.mean((preds.squeeze(-1) - tgt_y) ** 2, dim=0)
            uncertainty = torch.abs(preds)

        loss = torch.sqrt(loss)
        return loss, preds, uncertainty

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds, _, _, _ = self.forward(data_batch=data_batch, epoch=epoch,
                                                              use_second_order=self.args.second_order and
                                                                               epoch > self.args.first_order_to_second_order_epoch,
                                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                              num_steps=self.args.num_updates,
                                                              training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):

        losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all = self.forward(data_batch=data_batch,
                                                                                                 epoch=epoch,
                                                                                                 use_second_order=False,
                                                                                                 use_multi_step_loss_optimization=True,
                                                                                                 num_steps=self.args.test_num_updates,
                                                                                                 training_phase=False)

        return losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all

    def meta_update(self, loss):

        try:
            self.optimizer.zero_grad()
            loss.backward()
            if 'imagenet' in self.args.dataset_name:
                for name, param in self.regressor.named_parameters():
                    if param.requires_grad:
                        param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
            self.optimizer.step()
            if self.args.ddp:
                self.optimizer.sync_parameters()
        except:
            pass

    def run_weighted_train_iter(self, data_batch, epoch, cur_test_data, self_weight):
        epoch = int(epoch)
        # self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds, _, _, _ = self.forward(data_batch=data_batch, epoch=epoch,
                                                              use_second_order=self.args.second_order and
                                                                               epoch > self.args.first_order_to_second_order_epoch,
                                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                              num_steps=self.args.num_updates,
                                                              training_phase=True)

        losses_test, _, _, _, _ = self.forward(data_batch=cur_test_data, epoch=epoch,
                                               use_second_order=self.args.second_order and
                                                                epoch > self.args.first_order_to_second_order_epoch,
                                               use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                               num_steps=self.args.num_updates,
                                               training_phase=True,
                                               same_sup_tgt=True)

        self.meta_update(loss=losses['loss'] + self_weight * losses_test['loss'])  #
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, losses_test

    def run_train_iter(self, data_batch, epoch):

        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):

        if self.training:
            self.eval()

        losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all = self.evaluation_forward_prop(
            data_batch=data_batch,
            epoch=self.current_epoch)

        return losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all

    def run_validation_iter_knnmaml(self, data_batch, data_batch_train):
        if self.training:
            self.eval()
        losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all = self.forward_knn(data_batch=data_batch,
                                                                                                 data_batch_knn=data_batch_train,
                                                                                                 epoch=self.current_epoch,
                                                                                                 use_second_order=False,
                                                                                                 use_multi_step_loss_optimization=True,
                                                                                                 num_steps=self.args.test_num_updates,
                                                                                                 training_phase=False)

        return losses, per_task_target_preds, final_weights, sup_losses, uncertainty_all


    def run_many_shot_validation_iter(self, data_batch):

        self.many_shot_forward(data_batch)

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):

        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):

        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
