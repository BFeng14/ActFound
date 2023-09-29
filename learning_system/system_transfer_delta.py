import torch
import torch.nn as nn
from .system_base import RegressorBase

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg


class TransferDeltaRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super(TransferDeltaRegressor, self).__init__(input_shape, args)
        self.regressor = FCNReLUNormNetworkQSAR(input_shape=self.input_shape, args=self.args, meta=True).cuda()
        self.softmax = nn.Softmax(dim=0)
        self.post_init(args)

    def forward(self, data_batch, epoch, num_steps, is_training_phase, **kwargs):
        xs, ys, splits, assay_idxes, assay_weight, _ = data_batch

        total_losses = []
        per_task_target_preds = []
        self.regressor.zero_grad()
        final_weights = []
        per_task_metrics = []
        self.is_training_phase = is_training_phase

        for x_task, y_task, split, assay_idx in zip(xs, ys, splits, assay_idxes):
            y_task = y_task.float().cuda()
            x_task = x_task.float().cuda()

            names_weights_copy = None
            support_loss_each_step = None
            task_losses = []
            if not is_training_phase:
                names_weights_copy, support_loss_each_step, task_losses = self.inner_loop(x_task, y_task, assay_idx, split,
                                                                                          is_training_phase, epoch,
                                                                                          num_steps)
                target_loss, target_preds = self.net_forward(x=x_task,
                                                             y=y_task,
                                                             assay_idx=assay_idx,
                                                             split=split,
                                                             weights=names_weights_copy,
                                                             backup_running_statistics=False, training=True,
                                                             num_step=num_steps - 1,
                                                             is_support=False)
            else:
                target_loss, target_preds = self.net_forward(x=x_task,
                                                             y=y_task,
                                                             assay_idx=assay_idx,
                                                             split=split,
                                                             weights=names_weights_copy,
                                                             backup_running_statistics=False, training=True,
                                                             num_step=num_steps - 1,
                                                             is_support=False)
            task_losses.append(target_loss)

            per_task_target_preds.append(target_preds.detach().cpu().numpy())
            metrics = self.get_metric(y_task, target_preds, split)
            metrics["each_step_loss"] = support_loss_each_step
            per_task_metrics.append(metrics)
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            final_weights.append(names_weights_copy)

            if not is_training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   loss_weights=assay_weight)
        return losses, per_task_target_preds, final_weights, per_task_metrics

    def net_forward(self, x, y, split, weights, backup_running_statistics, training, num_step, assay_idx=None,
                    is_support=False):

        sup_idx = torch.nonzero(split)[:, 0]
        tgt_idx = torch.nonzero(1. - split)[:, 0]
        sup_x = x[sup_idx]
        tgt_x = x[tgt_idx]
        sup_y = y[sup_idx]
        tgt_y = y[tgt_idx]
        sup_num = torch.sum(split)
        tgt_num = split.shape[0] - sup_num

        out_embed, out_value = self.regressor.forward(x=x, params=weights,
                                                      training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)

        support_value = out_value[sup_idx]
        tgt_value = out_value[tgt_idx]
        support_features_flat = out_embed[sup_idx]
        query_features_flat = out_embed[tgt_idx]

        ddg_sup_std = sup_y.unsqueeze(-1) - sup_y.unsqueeze(0)
        if self.is_training_phase:
            rescale = 1.0
        else:
            rescale = max(0.2, sup_y.std())
        if is_support:
            ddg_pred = support_value.unsqueeze(-1) - support_value.unsqueeze(0)
            ddg_real = sup_y.unsqueeze(-1) - sup_y.unsqueeze(0)
            ddg_pred = ddg_pred * rescale

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
            loss = loss_dg / rescale**2
        else:
            ddg_pred_1 = support_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
            ddg_real_1 = sup_y.unsqueeze(-1) - tgt_y.unsqueeze(0)
            ddg_pred_2 = tgt_value.unsqueeze(-1) - tgt_value.unsqueeze(0)
            ddg_real_2 = tgt_y.unsqueeze(-1) - tgt_y.unsqueeze(0)

            cross_sim_mat = self.get_sim_matrix(sup_x, tgt_x)
            _, topk_idx = torch.topk(cross_sim_mat, dim=0, k=cross_sim_mat.shape[0] // 2)

            embed_sim_matrix = self.cossim_matrix(support_features_flat, query_features_flat) / self.temp
            sup_y_repeat = sup_y.unsqueeze(-1).repeat(1, tgt_num)  # [sup_num, tgt_num]
            preds_all = sup_y_repeat - ddg_pred_1 * rescale

            preds_select = torch.gather(preds_all, 0, topk_idx)
            embed_sim_matrix_select = torch.gather(embed_sim_matrix, 0, topk_idx)
            embed_sim_matrix_select = self.softmax(embed_sim_matrix_select)
            preds = torch.sum(preds_select * embed_sim_matrix_select, dim=0)

            tgt_sim_mat = self.get_sim_matrix(tgt_x, tgt_x) - torch.eye(tgt_num).cuda()
            _, tgt_topk_idx = torch.topk(tgt_sim_mat, dim=0, k=tgt_sim_mat.shape[0] // 2)

            loss_2 = self.robust_square_error(ddg_pred_2, ddg_real_2, tgt_topk_idx)
            loss_1 = self.robust_square_error(ddg_pred_1, ddg_real_1, topk_idx)
            loss_dg = torch.mean((preds - tgt_y) ** 2)
            loss = loss_2 * 0.25 + loss_1 * 0.5 + loss_dg

        loss = torch.sqrt(loss)
        return loss, preds
