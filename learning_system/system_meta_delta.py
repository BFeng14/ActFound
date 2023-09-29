import torch
import torch.nn as nn
import numpy as np
import json
from .system_base import RegressorBase

from models.meta_neural_network_architectures import FCNReLUNormNetworkQSAR, AssayFCNReLUNormNetworkReg


class MetaDeltaRegressor(RegressorBase):
    def __init__(self, input_shape, args):
        super(MetaDeltaRegressor, self).__init__(input_shape, args)
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
        data_batch_knn = kwargs.get("data_batch_knn", None)

        for x_task, y_task, split, assay_idx in zip(xs, ys, splits, assay_idxes):
            y_task = y_task.float().cuda()
            x_task = x_task.float().cuda()
            if data_batch_knn is None:
                names_weights_copy, support_loss_each_step, task_losses = self.inner_loop(x_task, y_task, assay_idx,
                                                                                          split,
                                                                                          is_training_phase, epoch,
                                                                                          num_steps)
            else:
                names_weights_copy, support_loss_each_step, task_losses = self.inner_loop_knn(data_batch_knn,
                                                                                              x_task, y_task, assay_idx,
                                                                                              split,
                                                                                              is_training_phase, epoch,
                                                                                              num_steps)

            target_loss, target_preds = self.net_forward(x=x_task,
                                                         y=y_task,
                                                         assay_idx=assay_idx,
                                                         split=split,
                                                         weights=names_weights_copy,
                                                         backup_running_statistics=False, training=True,
                                                         num_step=num_steps - 1)
            if not is_training_phase:
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

    def inner_loop_knn(self, data_batch_knn, x_task, y_task, assay_idx, split, is_training_phase, epoch, num_steps):
        xs_knn, ys_knn, splits_knn, _, assay_weight_knn, _ = data_batch_knn[0]
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
            support_loss_knns = []
            for x_task_knn, y_task_knn, split_knn, weight_knn in zip(xs_knn, ys_knn, splits_knn, assay_weight_knn):
                y_task_knn = y_task_knn.float().cuda()
                x_task_knn = x_task_knn.float().cuda()
                split_knn = (split_knn + 1) / (split_knn + 1)
                support_loss_knn, _ = self.net_forward(x=x_task_knn,
                                                       y=y_task_knn,
                                                       assay_idx=None,
                                                       split=split_knn.long(),
                                                       is_support=True,
                                                       weights=names_weights_copy,
                                                       backup_running_statistics=False,
                                                       training=True, num_step=num_step)
                support_loss_knns.append(support_loss_knn * weight_knn)

            support_loss_each_step.append(support_loss.detach().cpu().item())
            support_loss = support_loss + sum(support_loss_knns)
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

    def find_knn_batch(self, data_batch):
        def compute_dist(weight1, weight2):
            return torch.matmul(weight1, weight2.transpose(0, 1)) / (torch.linalg.norm(weight1, dim=-1).unsqueeze(-1) *
                                                                     torch.linalg.norm(weight2, dim=-1).unsqueeze(0))

        init_weight = self.get_init_weight()
        self.args.knn_maml = False
        loss, per_task_target_preds, final_weights, per_task_metrics = self.run_validation_iter(data_batch)
        self.args.knn_maml = True
        task_weight = final_weights[0]["layer_dict.linear.weights"]
        task_feat = task_weight - init_weight
        dists = compute_dist(self.train_assay_feat_all, task_feat).detach().cpu().numpy()
        assay_weights = dists[:, 0]
        knn_idx = np.nonzero(dists > self.args.knn_dist_thres)[0]

        topk_weights = assay_weights[knn_idx]
        knn_assay_idx = [self.train_assay_idxes[x] for x in knn_idx]
        if len(knn_assay_idx) > 0:
            top16_idx = np.argsort(-topk_weights)[:16]
            top16_weight = topk_weights[top16_idx]
            knn_assay_idx = [knn_assay_idx[idx] for idx in top16_idx]
            data_batch_knn = [x for x in self.dataloader.get_train_batches_weighted(top16_weight * 0.2, knn_assay_idx, len(knn_assay_idx))]
        else:
            data_batch_knn = []
        return data_batch_knn, loss, per_task_target_preds, final_weights, per_task_metrics

    def prepare_knn_maml(self, dataloader):
        self.dataloader = dataloader
        self.train_assay_feat_all = torch.tensor(np.load(self.args.train_assay_feat_all)).cuda()
        train_assay_names_all = json.load(open(self.args.train_assay_idxes, "r"))
        assay_names = dataloader.dataset.assaes
        assay_names_dict = {x: i for i, x in enumerate(assay_names)}
        self.train_assay_idxes = [assay_names_dict[assay_name] for assay_name in train_assay_names_all]

    def run_validation_iter_knnmaml(self, data_batch):
        if self.training:
            self.eval()
        data_batch_knn, losses, per_task_target_preds, final_weights, per_task_metrics = self.find_knn_batch(data_batch)
        split = data_batch[2][0]
        sup_num = torch.sum(split)
        if len(data_batch_knn) > 0 and sup_num >= 12:
            # print("before", per_task_metrics[0]["r2"])
            losses, per_task_target_preds, final_weights, per_task_metrics = self.forward(data_batch=data_batch,
                                                                                        data_batch_knn=data_batch_knn,
                                                                                        epoch=self.current_epoch,
                                                                                        num_steps=self.args.test_num_updates,
                                                                                        is_training_phase=False)
            # print("after", per_task_metrics[0]["r2"])
            per_task_metrics[0]["knn_maml"] = True
        return losses, per_task_target_preds, final_weights, per_task_metrics

    def run_validation_iter(self, data_batch):
        if self.args.knn_maml:
            return self.run_validation_iter_knnmaml(data_batch)
        else:
            return super().run_validation_iter(data_batch)

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
        if epoch >= self.args.begin_lrloss_epoch:
            kname = "layer_dict.linear.weights".replace(".", "-")
            lr_param = self.inner_loop_optimizer.names_learning_rates_dict[kname]
            losses['loss'] = losses['loss'] + torch.sum(lr_param * lr_param) * self.args.lrloss_weight
        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds