import torch
import torch.nn as nn
from .system_base import RegressorBase

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg



class ProtoNetRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super(ProtoNetRegressor, self).__init__(input_shape, args)
        self.regressor = AssayFCNReLUNormNetworkReg(input_shape=self.input_shape, args=self.args, meta=True).cuda()
        self.softmax = nn.Softmax(dim=0)
        self.post_init(args)

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

            target_loss, target_preds = self.net_forward(x=x_task,
                                                         y=y_task,
                                                         assay_idx=assay_idx,
                                                         split=split,
                                                         weights=None,
                                                         backup_running_statistics=False, training=True,
                                                         num_step=num_steps - 1)

            per_task_target_preds.append(target_preds.detach().cpu().numpy())
            total_losses.append(target_loss)
            final_weights.append(None)
            sup_losses.append(None)

            if not is_training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)
        return losses, per_task_target_preds, final_weights, sup_losses

    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False):
        out = self.regressor.forward(x=x, split=split, params=weights,
                                     training=training,
                                     backup_running_statistics=backup_running_statistics,
                                     num_step=num_step)
        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        support_features_flat = out[sup_idx]
        query_features_flat = out[tgt_idx]
        sup_y = y[sup_idx]
        tgt_y = y[tgt_idx]

        if is_support:
            sim_matrix = self.cossim_matrix(support_features_flat, support_features_flat) / self.temp
            sim_matrix = sim_matrix - 1e9 * torch.eye(support_features_flat.shape[0]).cuda()
            sim_matrix = self.softmax(sim_matrix)
            preds = torch.matmul(sup_y, sim_matrix)
            loss = torch.mean((preds.squeeze(-1) - sup_y) ** 2, dim=0)
            if torch.isnan(loss):
                print("loss is nan:", sim_matrix, sup_y)
        else:
            sim_matrix = self.cossim_matrix(support_features_flat, query_features_flat) / self.temp
            sim_matrix = self.softmax(sim_matrix)
            preds = torch.matmul(sup_y, sim_matrix)
            loss = torch.mean((preds.squeeze(-1) - tgt_y) ** 2, dim=0)

        loss = torch.sqrt(loss)
        return loss, preds