import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

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


class RegressorBase(nn.Module):
    def __init__(self, input_shape, args):

        super(RegressorBase, self).__init__()
        self.args = args
        self.batch_size = args.meta_batch_size
        self.current_epoch = 0
        self.input_shape = input_shape
        self.rng = set_torch_seed(seed=0)

        self.args.rng = self.rng

    def post_init(self, args):
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(args=args,
                                                                    init_learning_rate=args.update_lr,
                                                                    total_num_inner_loop_steps=self.args.num_updates,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.regressor.named_parameters()))
        self.temp = 0.2
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape, value.requires_grad)

        self.cuda()
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.requires_grad)

        opt = optim.Adam(self.trainable_parameters(), lr=args.meta_lr, amsgrad=False)
        self.optimizer = opt
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=opt,
                                                              T_max=self.args.metatrain_iterations,
                                                              eta_min=self.args.min_learning_rate)

    def cossim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

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
                if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                    yield param

    def get_inner_loop_parameter_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                    param_dict[name] = param.cuda()

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx, sup_number):
        self.regressor.zero_grad(names_weights_copy)

        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                        create_graph=use_second_order)

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
        init_weight = init_weight["layer_dict.linear.weights"]
        return init_weight

    def inner_loop(self, x_task, y_task, assay_idx, split, is_training_phase, epoch, num_steps):
        task_losses = []
        per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
        support_loss_each_step = []
        sup_num = torch.sum(split)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())
        use_second_order = self.args.second_order and epoch > self.args.first_order_to_second_order_epoch
        for num_step in range(num_steps):
            support_loss, support_preds = self.net_forward(x=x_task,
                                                           y=y_task,
                                                           assay_idx=assay_idx,
                                                           split=split,
                                                           is_support=True,
                                                           weights=names_weights_copy,
                                                           backup_running_statistics=
                                                           True if (num_step == 0) else False,
                                                           training=True, num_step=num_step)
            support_loss_each_step.append(support_loss.detach().cpu().item())
            if support_loss >= support_loss_each_step[0] * 10 or sup_num <= 5:
                pass
            else:
                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order if is_training_phase else False,
                                                                  current_step_idx=num_step,
                                                                  sup_number=torch.sum(split))

            if is_training_phase:
                is_multi_step_optimize = self.args.use_multi_step_loss_optimization and epoch < self.args.multi_step_loss_num_epochs
                is_last_step = num_step == (self.args.num_updates - 1)
                if is_multi_step_optimize or is_last_step:
                    target_loss, target_preds = self.net_forward(x=x_task,
                                                                 y=y_task,
                                                                 assay_idx=assay_idx,
                                                                 split=split,
                                                                 weights=names_weights_copy,
                                                                 backup_running_statistics=False,
                                                                 training=True,
                                                                 num_step=num_step)
                    if is_multi_step_optimize:
                        task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                    elif is_last_step:
                        task_losses.append(target_loss)
                        
        return names_weights_copy, support_loss_each_step, task_losses

    def get_metric(self, y, y_pred, split):
        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        y_true = y[tgt_idx]
        y_train_mean = torch.mean(y[sup_idx]).cpu().item()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        r2 = np.corrcoef(y_true, y_pred)[0, 1]
        if math.isnan(r2) or r2 < 0.:
            r2 = 0.
        else:
            r2 = r2 ** 2

        numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((y_true - y_train_mean) ** 2).sum(axis=0, dtype=np.float64)
        if denominator == 0:
            R2os = 0.0
        else:
            R2os = 1.0 - (numerator / denominator)
        return {"r2": float(r2), "rmse": float(rmse), "R2os": float(R2os), "y_train_mean": y_train_mean,
                "pred": [float(x) for x in y_pred], "ture": [float(x) for x in y_true]}

    def forward(self, data_batch, epoch, num_steps, is_training_phase, **kwargs):
        raise NotImplementedError

    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False, **kwargs):
        raise NotImplementedError

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def meta_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds, _, _ = self.forward(data_batch=data_batch, epoch=epoch,
                                                           num_steps=self.args.num_updates,
                                                           is_training_phase=True)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        if self.training:
            self.eval()
        losses, per_task_target_preds, final_weights, per_task_metrics = self.forward(data_batch=data_batch,
                                                                                epoch=self.current_epoch,
                                                                                num_steps=self.args.test_num_updates,
                                                                                is_training_phase=False)
        return losses, per_task_target_preds, final_weights, per_task_metrics

    def save_model(self, model_save_dir, state):
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):

        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
