import torch
import torch.nn as nn
from .system_base import RegressorBase

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg


class MAMLRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super(MAMLRegressor, self).__init__(input_shape, args)
        self.regressor = FCNReLUNormNetworkQSAR(input_shape=self.input_shape, args=self.args, meta=True).cuda()
        self.post_init(args)

    def cossim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, data_batch, epoch, num_steps, is_training_phase):
        xs, ys, splits, assay_idxes, assay_weight, _ = data_batch

        total_losses = []
        per_task_target_preds = []
        self.regressor.zero_grad()
        final_weights = []
        sup_losses = []

        for x_task, y_task, split, assay_idx in zip(xs, ys, splits, assay_idxes):
            y_task = y_task.float().cuda()
            x_task = x_task.float().cuda()
            names_weights_copy, support_loss_each_step, task_losses = self.inner_loop(x_task, y_task, assay_idx, split,
                                                                                      is_training_phase, epoch, num_steps)

            target_preds = None
            if not is_training_phase:
                target_loss, target_preds = self.net_forward(x=x_task,
                                                             y=y_task,
                                                             assay_idx=assay_idx,
                                                             split=split,
                                                             weights=names_weights_copy,
                                                             backup_running_statistics=False, training=True,
                                                             num_step=num_steps - 1)
                task_losses.append(target_loss)

            per_task_target_preds.append(target_preds.detach().cpu().numpy())
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            final_weights.append(names_weights_copy)
            sup_losses.append(support_loss_each_step)

            if not is_training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)
        return losses, per_task_target_preds, final_weights, sup_losses


    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False):

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

        if is_support:
            preds = support_value
            loss = torch.mean((preds - sup_y) ** 2)
        else:
            preds = tgt_value
            loss = torch.mean((preds - tgt_y) ** 2)

        loss = torch.sqrt(loss)
        return loss, preds
