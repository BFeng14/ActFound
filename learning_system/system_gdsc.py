import torch
import torch.nn as nn
from .system_base import RegressorBase

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg



class GDSCRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super(GDSCRegressor, self).__init__(input_shape, args)
        import pickle
        self.cellline_feats = pickle.load(open(self.args.cell_line_feat, "rb"))
        self.regressor = FCNReLUNormNetworkQSAR(input_shape=self.input_shape, args=self.args, meta=True).cuda()
        self.post_init(args)

    def forward(self, data_batch, epoch, num_steps, is_training_phase):
        xs, ys, splits, assay_idxes, assay_weight, _ = data_batch

        total_losses = []
        per_task_target_preds = []
        per_task_metrics = []
        self.regressor.zero_grad()
        final_weights = []

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
            per_task_metrics.append(self.get_metric(y_task, target_preds, split))
            total_losses.append(target_loss)
            final_weights.append(None)

            if not is_training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)
        return losses, per_task_target_preds, final_weights, per_task_metrics

    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False):
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
        return loss_dg, tgt_value