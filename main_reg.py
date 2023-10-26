import argparse
import copy
import random

import numpy as np
import torch
import os
import math
import json

from tqdm import tqdm
from dataset import dataset_constructor
from learning_system import system_selector

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', default='chembl', type=str)
    parser.add_argument('--model_name', default='actfound', type=str)
    parser.add_argument('--dim_w', default=2048, type=int, help='dimension of w')
    parser.add_argument('--hid_dim', default=2048, type=int, help='dimension of w')
    parser.add_argument('--num_stages', default=2, type=int, help='num stages')
    parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
    parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
    parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
    parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
    parser.add_argument('--second_order', default=1, type=int, help='second_order')
    parser.add_argument('--first_order_to_second_order_epoch', default=10, type=int, help='first_order_to_second_order_epoch')

    parser.add_argument('--transfer_lr', default=0.004, type=float,  help='transfer_lr')
    parser.add_argument('--test_sup_num', default="0", type=str)
    parser.add_argument('--test_repeat_num', default=10, type=int)

    parser.add_argument('--test_write_file', default="./test_result_debug/", type=str)
    parser.add_argument('--expert_test', default="", type=str)
    parser.add_argument('--scaffold_split', default=False, action='store_true')

    parser.add_argument('--train_seed', default=1111, type=int, help='train_seed')
    parser.add_argument('--val_seed', default=1111, type=int, help='val_seed')
    parser.add_argument('--test_seed', default=1111, type=int, help='test_seed')

    parser.add_argument('--metatrain_iterations', default=80, type=int,
                        help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
    parser.add_argument('--meta_batch_size', default=16, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--min_learning_rate', default=0.0001, type=float, help='min_learning_rate')
    parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.00015, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
    parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')


    ## Logging, saving, and testing options
    parser.add_argument('--logdir', default='', type=str,
                        help='directory for summaries and checkpoints.')
    parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

    parser.add_argument('--new_ddg', default=False, action='store_true')

    parser.add_argument('--input_celline', default=False, action='store_true')
    parser.add_argument('--cell_line_feat', default='./datas/gdsc/cellline_to_feat.pkl')
    parser.add_argument('--cross_test', default=False, action='store_true')
    parser.add_argument('--gdsc_pretrain_data', default="none", type=str)
    parser.add_argument('--use_byhand_lr', default=False, action='store_true')

    parser.add_argument('--inverse_ylabel', default=False, action='store_true')
    parser.add_argument('--knn_maml', default=False, action='store_true')
    parser.add_argument('--train_assay_feat_all', default='')
    parser.add_argument('--train_assay_idxes', default='')
    parser.add_argument('--knn_dist_thres', default=0.3, type=float)
    parser.add_argument('--begin_lrloss_epoch', default=50, type=int)
    parser.add_argument('--lrloss_weight', default=35., type=float)
    return parser


def train(args, model, dataloader):
    Print_Iter = 200

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    begin_epoch = 0
    if args.resume == 1:
        begin_epoch = args.test_epoch + 1
    if args.datasource == "gdsc":
        _, last_test_result = test(args, begin_epoch, model, dataloader, is_test=True)
    else:
        _, last_test_result = test(args, begin_epoch, model, dataloader, is_test=False)

    beat_epoch = -1
    print_loss = 0.0
    print_step = 0
    for epoch in range(begin_epoch, args.metatrain_iterations):
        train_data_all = dataloader.get_train_batches()
        for step, cur_data in enumerate(train_data_all):
            meta_batch_loss, _ = model.run_train_iter(cur_data, epoch)

            if (print_step+1) % Print_Iter == 0 or step == len(train_data_all)-1:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss/print_step))
                print_loss = 0.0
                print_step = 0
            else:
                print_loss += meta_batch_loss['loss']
                print_step += 1

        if args.datasource == "gdsc":
            if not (epoch-begin_epoch+1) % 5 == 0:
                continue
            res_dict, test_result = test(args, epoch, model, dataloader, is_test=True)
            # torch.save(model.state_dict(), './scripts/gdsc/checkpoints/model_{}'.format(epoch))
            if last_test_result < test_result:
                last_test_result = test_result
                write_dir = os.path.join(args.test_write_file, "on_"+args.gdsc_pretrain_data)
                if not os.path.exists(write_dir):
                    os.system(f"mkdir -p {write_dir}")
                json.dump(res_dict, open(os.path.join(write_dir, f"sup_num_{args.test_sup_num}.json"), "w"))
                print(f"saving best result into {write_dir}")
        else:
            _, test_result = test(args, epoch, model, dataloader, is_test=False)
            torch.save(model.state_dict(), '{0}/{2}/model_{1}'.format(args.logdir, epoch, exp_string))
            if last_test_result < test_result:
                last_test_result = test_result
                beat_epoch = epoch
                torch.save(model.state_dict(), '{0}/{2}/model_best'.format(args.logdir, epoch, exp_string))
            print("beat valid epoch is:", beat_epoch)


def test(args, epoch, model, dataloader, is_test=True):
    kname = "layer_dict.linear.weights".replace(".", "-")
    print(model.inner_loop_optimizer.names_learning_rates_dict[kname])
    cir_num = args.test_repeat_num
    r2_list = []
    R2os_list = []
    rmse_list = []
    res_dict = {}
    print(cir_num)
    for cir in range(cir_num):
        if is_test:
            test_data_all = dataloader.get_test_batches(repeat_cnt=cir)
        else:
            test_data_all = dataloader.get_val_batches(repeat_cnt=cir)
        if args.knn_maml:
            test_data_all = [x for x in test_data_all]
        for step, cur_data in enumerate(test_data_all):
            ligands_x = cur_data[0][0]
            if len(ligands_x) <= args.test_sup_num:
                continue
            losses, per_task_target_preds, final_weights, per_task_metrics = model.run_validation_iter(cur_data)
            r2_list.append(per_task_metrics[0]["r2"])
            R2os_list.append(per_task_metrics[0]["R2os"])
            rmse_list.append(per_task_metrics[0]["rmse"])
            assay_name = cur_data[3][0]
            if assay_name not in res_dict.keys():
                res_dict[assay_name] = []
            res_dict[assay_name].append(per_task_metrics[0])

    rmse_i = np.mean(rmse_list)
    median_r2 = np.median(r2_list, 0)
    mean_r2 = np.mean(r2_list, 0)
    valid_cnt = len([x for x in r2_list if x > 0.3])
    print(
        'epoch is: {}, mean rmse is: {:.3f}'.
        format(epoch, rmse_i))
    print(
        'epoch is: {}, r2: mean is: {:.3f}, median is: {:.3f}, cnt>0.3 is: {:.3f}'.
        format(epoch, mean_r2, median_r2, valid_cnt))
    median_r2os = np.median(R2os_list, 0)
    mean_r2os = np.mean(R2os_list, 0)
    valid_cnt = len([x for x in R2os_list if x > 0.3])
    print(
        'epoch is: {}, R2os: mean is: {:.3f}, median is: {:.3f}, cnt>0.3 is: {:.3f}'.
        format(epoch, mean_r2os, median_r2os, valid_cnt))
    return res_dict, mean_r2-rmse_i+1


def prepare_assay_feat(args, model, dataloader):
    if os.path.exists(args.train_assay_feat_all):
        return
    train_data_all = dataloader.get_train_batches_weighted()
    init_weight = model.get_init_weight().detach().cpu().numpy().squeeze()
    train_weights_all = []
    train_assay_names_all = []
    loss_all = []
    for train_idx, cur_data in tqdm(enumerate(train_data_all)):
        train_assay_names_all.append(cur_data[3][0])
        loss, _, final_weights, _ = model.run_validation_iter(cur_data)
        loss_all.append(loss['loss'].detach().cpu().item())
        task_weight = final_weights[0]["layer_dict.linear.weights"].detach().cpu().numpy().squeeze()
        task_feat = task_weight - init_weight
        train_weights_all.append(task_feat)
        if (train_idx+1)%200 == 0:
            print(sum(loss_all) / len(loss_all))

    print(sum(loss_all)/len(loss_all))
    np.save(args.train_assay_feat_all, np.array(train_weights_all))
    json.dump(train_assay_names_all, open(args.train_assay_idxes, "w"))


def main():
    model = system_selector(args)(args=args, input_shape=(2, args.dim_w))
    dataloader = dataset_constructor(args)

    if args.train == 1:
        if args.resume == 1:
            model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
            if not os.path.exists(model_file):
                model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
            print("resume training from", model_file)
            try:
                model.load_state_dict(torch.load(model_file))
            except Exception as e:
                model.load_state_dict(torch.load(model_file), strict=False)
        train(args, model, dataloader)
    elif args.train == 0:
        args.meta_batch_size = 1
        model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
        if not os.path.exists(model_file):
            model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
        if not isinstance(args.test_sup_num, list):
            args.test_sup_num = [args.test_sup_num]
        test_sup_num_all = copy.deepcopy(args.test_sup_num)
        for test_sup_num in test_sup_num_all:
            args.test_sup_num = test_sup_num
            try:
                model.load_state_dict(torch.load(model_file))
            except Exception as e:
                print(e)
                model.load_state_dict(torch.load(model_file), strict=False)
            if args.knn_maml:
                old_sup_num = args.test_sup_num
                args.test_sup_num = 0.90
                args.knn_maml = False
                prepare_assay_feat(args, model, dataloader)
                args.test_sup_num = old_sup_num
                args.knn_maml = True
                model.prepare_knn_maml(dataloader)
            res_dict, _ = test(args, args.test_epoch, model, dataloader)
            write_dir = os.path.join(args.test_write_file, args.model_name)
            if not os.path.exists(write_dir):
                os.system(f"mkdir -p {write_dir}")
            print("write result to", f"{write_dir}/sup_num_{args.test_sup_num}.json")
            print("\n\n\n\n")
            json.dump(res_dict, open(os.path.join(write_dir, f"sup_num_{args.test_sup_num}.json"), "w"))
    elif args.train == 2:
        test_data_all = dataloader.get_test_batches()
        model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
        if not os.path.exists(model_file):
            model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
        try:
            model.load_state_dict(torch.load(model_file))
        except Exception as e:
            print(e)
            model.load_state_dict(torch.load(model_file), strict=False)

        save_dir = "./ligand_feats/ligands_bdb_indomain_test_meta_delta"
        os.system(f"mkdir -p {save_dir}")
        for train_idx, cur_data in tqdm(enumerate(test_data_all)):
            ligand_num = len(cur_data[1][0])
            x_task = cur_data[0][0]
            y_task = cur_data[1][0]
            smiles = cur_data[-1][0]
            x_task = x_task.float().cuda()
            assay_name = cur_data[3][0].replace("/", "_")
            feat, _ = model.regressor.forward_feat(x=x_task, num_step=0)
            feat = feat.detach()
            # pred_y = pred_y.detach().cpu().numpy()
            # r2 = np.corrcoef(y_task, pred_y)[0, 1]
            # print(assay_name, r2)
            np.save(f"{save_dir}/{assay_name}", feat.cpu().numpy())
            json.dump(smiles, open(f"{save_dir}/{assay_name}_smiles.json", "w"))
        exit()


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    if "transfer" in args.model_name.lower() or "protonet" in args.model_name.lower():
        args.per_step_bn_statistics = False
    print(args)

    try:
        args.test_sup_num = json.loads(args.test_sup_num)
    except:
        args.test_sup_num = float(args.test_sup_num)
        if args.test_sup_num > 1:
            args.test_sup_num = int(args.test_sup_num)

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    random.seed(1)
    np.random.seed(2)

    datasource = args.datasource
    if args.datasource == "gdsc":
        datasource = args.gdsc_pretrain_data
    exp_string = f'data_{datasource}.mbs_{args.meta_batch_size}.metalr_0.00015.innerlr_{args.update_lr}'
    main()
