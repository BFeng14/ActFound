import argparse
import copy
import random

import numpy as np
import torch
import os
import math
import json

from tqdm import tqdm
from datas.data_chembl_Assay_reg import CHEMBLMetaLearningSystemDataLoader
from datas.data_gdsc_reg import GDSCMetaLearningSystemDataLoader
from datas.data_expert_reg import EXPERTMetaLearningSystemDataLoader
from datas.data_pqsar_Assay_reg import PQSARMetaLearningSystemDataLoader
from datas.data_fsmol_Assay_reg import FsmolMetaLearningSystemDataLoader
from learning_system.system_meta_delta import MAMLRegressor

def get_args():
    parser = argparse.ArgumentParser(description='MetaMix')
    parser.add_argument('--datasource', default='drug', type=str, help='drug')
    parser.add_argument('--dim_w', default=1024, type=int, help='dimension of w')
    parser.add_argument('--hid_dim', default=500, type=int, help='dimension of w')
    parser.add_argument('--num_stages', default=2, type=int, help='num stages')
    parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
    parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
    parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
    parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
    parser.add_argument('--second_order', default=1, type=int, help='second_order')
    parser.add_argument('--first_order_to_second_order_epoch', default=10, type=int, help='first_order_to_second_order_epoch')
    parser.add_argument('--mixup', default=False, action='store_true', help='metamix')
    parser.add_argument('--multitask', default=False, action='store_true', help='multitask')
    parser.add_argument('--assay_desc', default=False, action='store_true', help='assay_desc')
    parser.add_argument('--transfer_l', default=False, action='store_true', help='transfer_l')
    parser.add_argument('--transfer_lr', default=0.01, type=float,  help='transfer_lr')
    parser.add_argument('--assay_feat_dim', default=768, type=int, help='assay_feat_dim')
    parser.add_argument('--ddg_clip_value', default=3., type=float, help='ddg_clip_value')
    parser.add_argument('--sim_thres', default=0.2, type=float, help='sim_thres')
    parser.add_argument('--weight_k', default=160, type=int)
    parser.add_argument('--ddp', default=False, action='store_true')
    parser.add_argument('--test_sup_num', default="16", type=str)
    parser.add_argument('--test_write_file', default="./test_result_debug/", type=str)
    


    parser.add_argument('--vinafp', default=False, action='store_true', help='vina interaction fp')
    parser.add_argument('--qsar', default=False, action='store_true', help='qsar')
    parser.add_argument('--normal_ddg', default=False, action='store_true')
    parser.add_argument('--normal_loss', default=False, action='store_true')
    parser.add_argument('--use_gnn', default=False, action='store_true')
    parser.add_argument('--chembl', default=False, action='store_true')
    parser.add_argument('--expert_test', default="", type=str)
    parser.add_argument('--scaffold_split', default=False, action='store_true')
    parser.add_argument('--max_pair', default=100000, type=int, help='max_pair')
    parser.add_argument('--no_pair_weight', default=False, action='store_true')
    parser.add_argument('--use_vampire', default=False, action='store_true', help='use_vampire')
    parser.add_argument('--loss_kl_weight', default=0.5*1e-4, type=float, help='loss_kl_weight')


    parser.add_argument('--dim_y', default=1, type=int, help='dimension of w')
    parser.add_argument('--dataset_name', default='assay', type=str,
                        help='dataset_name.')
    # parser.add_argument('--dataset_path', default='ci9b00375_si_002.txt', type=str,
    #                     help='dataset_path.')
    # parser.add_argument('--type_filename', default='ci9b00375_si_001.txt', type=str,
    #                     help='type_filename.')
    # parser.add_argument('--compound_filename', default='ci9b00375_si_003.txt', type=str,
    #                     help='Directory of data files.')
    parser.add_argument('--experiment_file', default='/home/fengbin/BindingDB/BDB_per_target_pureligand/experiment_polymer.pkl', type=str,
                        help='Directory of data files.')
    parser.add_argument('--pair_file', default='/home/fengbin/BindingDB/BDB_per_target_pureligand/pair_polymer.pkl', type=str,
                        help='Directory of data files.')

    parser.add_argument('--fp_file', default='/home/fengbin/DDG_FB/scripts/data_curate/ligand_fp_dict.pkl', type=str,
                        help='fp_filename.')

    parser.add_argument('--target_assay_list', default='591252', type=str,
                        help='target_assay_list')

    parser.add_argument('--train_seed', default=1111, type=int, help='train_seed')
    parser.add_argument('--val_seed', default=1111, type=int, help='val_seed')
    parser.add_argument('--test_seed', default=1111, type=int, help='test_seed')
    parser.add_argument('--with_cellviab', default=False, action='store_true')

    parser.add_argument('--train_val_split', default=[0.9588, 0.0177736202, 0.023386342], type=list, help='train_val_split')
    parser.add_argument('--num_evaluation_tasks', default=100, type=int, help='num_evaluation_tasks')
    parser.add_argument('--drug_group', default=-1, type=int, help='drug group')
    parser.add_argument('--metatrain_iterations', default=50, type=int,
                        help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
    parser.add_argument('--meta_batch_size', default=4, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--min_learning_rate', default=0.0001, type=float, help='min_learning_rate')
    parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
    parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')
    parser.add_argument('--fusion_method', default='two', type=str, help='fusion_method')
    parser.add_argument('--num_assays', default=18000, type=int, help='num_assays')


    ## Logging, saving, and testing options
    parser.add_argument('--logdir', default='/home/fengbin/QSAR/MetaMix/Drug_ddg/checkpoint_BDB', type=str,
                        help='directory for summaries and checkpoints.')
    parser.add_argument('--datadir', default='/home/fengbin/BindingDB/BDB_per_target_pureligand/polymer', type=str, help='directory for datasets.')
    parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
    parser.add_argument('--trial', default=0, type=int, help='trial for each layer')

    parser.add_argument('--new_ddg', default=False, action='store_true')
    parser.add_argument('--cluster_meta', default=False, action='store_true')
    parser.add_argument('--nobackbone', default=False, action='store_true')

    parser.add_argument('--num_clusters', default=10, type=int)
    parser.add_argument('--test_repeat_num', default=10, type=int)
    parser.add_argument('--assay_idx_2_cls', default='./scripts/t-sne/chembl_indomain_trainset/assay_idx_2_cls_10.pkl', type=str,
                        help='directory for datasets.')
    parser.add_argument('--cls_model_file', default='/home/fengbin/meta_delta/checkpoints/checkpoint_chembl_reg_ddg_attn_newsplit/MetaMix.data_drug.mbs_16.metalr_0.00015.innerlr_0.001.drug_group_1.trial1/model_43', type=str,
                        help='directory for datasets.')
    parser.add_argument('--cluster_center', default='./scripts/t-sne/chembl_indomain_trainset/cluster_center.npy', type=str,
                        help='directory for datasets.')
    parser.add_argument('--cluster_load', default=False, action='store_true')
    parser.add_argument('--input_celline', default=False, action='store_true')
    parser.add_argument('--cell_line_feat', default='./scripts/gdsc/cellline_to_feat_pca.pkl')
    parser.add_argument('--cross_test', default=False, action='store_true')
    parser.add_argument('--use_byhand_lr', default=False, action='store_true')
    return parser


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    if args.transfer_l:
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

    if args.datasource not in ['drug', 'bdb', 'fsmol', 'pqsar']:
        datasource = 'bdb'
    else:
        datasource = args.datasource
    exp_string = f'MetaMix.data_{datasource}.mbs_{args.meta_batch_size}.metalr_{args.meta_lr}.innerlr_{args.update_lr}.drug_group_{args.drug_group}'

    if args.trial > 0:
        exp_string += '.trial{}'.format(args.trial)
    if args.mixup:
        exp_string += '.mix'

    print(exp_string)

def train(args, maml, dataloader):
    Print_Iter = 200
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    # test_valid = args.datasource in ["drug", "bdb"]
    testdG(args, 0, maml, dataloader, is_test=False)
    begin_epoch = 0
    if args.resume == 1:
        begin_epoch = args.test_epoch + 1
    last_test_result = -100000
    beat_epoch = -1
    for epoch in range(begin_epoch, args.metatrain_iterations):
        train_data_all = dataloader.get_train_batches()
        print_loss = 0.0
        print_step = 1e-6
        for step, cur_data in enumerate(train_data_all):
            meta_batch_loss, _ = maml.run_train_iter(cur_data, epoch)

            if step != 0 and step % Print_Iter == 0 or step == len(train_data_all)-1:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss/print_step))

                print_loss = 0.0
                print_step = 1e-6
            else:
                print_loss += meta_batch_loss['loss']
                print_step += 1


        if args.datasource in ["drug", "bdb", "pqsar", "fsmol"]:
            _, test_result = testdG(args, epoch, maml, dataloader, is_test=False)
            torch.save(maml.state_dict(), '{0}/{2}/model_{1}'.format(args.logdir, epoch, exp_string))
            if last_test_result < test_result:
                last_test_result = test_result
                beat_epoch = epoch
                torch.save(maml.state_dict(), '{0}/{2}/model_best'.format(args.logdir, epoch, exp_string))
            print("beat valid epoch is:", beat_epoch)
        else:
            if not (epoch-begin_epoch+1) % 5 == 0:
                continue
            res_dict, test_result = testdG(args, epoch, maml, dataloader, is_test=False)
            # torch.save(maml.state_dict(), './scripts/gdsc/checkpoints/model_{}'.format(epoch))
            if last_test_result < test_result:
                last_test_result = test_result
                write_dir = args.test_write_file
                if not os.path.exists(write_dir):
                    os.system(f"mkdir -p {write_dir}")
                json.dump(res_dict, open(os.path.join(write_dir, f"sup_num_{args.test_sup_num}.json"), "w"))


def knn_meta_learning(args, maml, dataloader, model_file):
    train_weights_all = np.load("./scripts/t-sne/chembl_indomain_trainset/lastlayer_weights.npy")
    train_assay_names_all = json.load(open("./scripts/t-sne/chembl_indomain_trainset/assay_ids.json", "r"))
    assay_names = dataloader.dataset.assaes
    assay_names_dict = {x: i for i, x in enumerate(assay_names)}
    train_assay_idxes = [assay_names_dict[assay_name] for assay_name in train_assay_names_all]

    init_weight = maml.get_init_weight()
    test_data_all = list(dataloader.get_test_batches())
    test_weights_all = []
    for test_idx, cur_data in tqdm(enumerate(test_data_all)):
        loss, _, final_weights, _, _ = maml.run_validation_iter(cur_data)
        if args.use_vampire:
            task_weight = final_weights[0]["layer_dict.vampire.mean"].detach().cpu().numpy().squeeze()
        else:
            task_weight = final_weights[0]["layer_dict.linear.weights"].detach().cpu().numpy().squeeze()
        task_feat = task_weight - init_weight
        test_weights_all.append(task_feat)

    def compute_dist(weight_list1, weight_list2):
        weight1 = np.array(weight_list1)
        weight2 = np.array(weight_list2)
        return np.dot(weight1, weight2.transpose()) / (np.expand_dims(np.linalg.norm(weight1, axis=-1), axis=-1) *
                                                       np.expand_dims(np.linalg.norm(weight2, axis=-1), axis=0) )

    # train_weights_all = copy.deepcopy(test_weights_all)
    B = compute_dist(train_weights_all, test_weights_all)

    from torch import optim
    res_dict = {}
    k = args.weight_k
    lr = args.meta_lr

    befores = []
    afters = []
    for test_idx, cur_test_data in enumerate(test_data_all):
        assay_weights = B[:, test_idx]
        topk_idx = np.nonzero(assay_weights > 0.3)[0]

        if len(topk_idx) == 0:
            continue

        assay_name = cur_test_data[3][0]
        print(assay_name)
        res_dict[assay_name] = {"topk_weights": None, "res": []}
        res = testdG_single_assay(maml, cur_test_data, None)
        befores.append(res[0])
        res_dict[assay_name]["res"].append(res)



        topk_weights = assay_weights[topk_idx]
        res_dict[assay_name]["topk_weights"] = topk_weights
        # print(topk_weights)
        topk_dataset_idx = [train_assay_idxes[x] for x in topk_idx]
        weighted_train_data_all = next(dataloader.get_train_batches_weighted(topk_weights*0.2, topk_dataset_idx, len(topk_idx), True))
        # print(weighted_train_data_all[4])
        res = testdG_single_assay(maml, cur_test_data, weighted_train_data_all)
        afters.append(res[0])
    print(np.mean(befores), np.mean(afters), len(befores))


def testdG_single_assay(maml, cur_test_data, weighted_train_data_all=None):
    if weighted_train_data_all is None:
        loss, per_task_target_preds, _, _, uncertainty_all = maml.run_validation_iter(cur_test_data)
    else:
        loss, per_task_target_preds, _, _, uncertainty_all = maml.run_validation_iter_knnmaml(cur_test_data, weighted_train_data_all)
    y = cur_test_data[1][0].numpy()  # 4:data format, 0:a list, 0:task
    split = cur_test_data[2][0].numpy()
    y_train = y[np.nonzero(split)]
    std_y_train = max(0.2, y_train.std())
    mean_y_train = y_train.mean()
    y_std = (y - mean_y_train) / std_y_train

    pred_res_std = np.squeeze(per_task_target_preds[0])

    tgt_idx = np.nonzero(1. - split)
    sup_idx = np.nonzero(split)
    sup_y_mean = np.mean(y_std[sup_idx])
    true_res_std = y_std[tgt_idx]
    r2, r2_new = r2_score_os(true_res_std, pred_res_std, sup_y_mean)

    pred_res = pred_res_std * std_y_train + mean_y_train
    true_res = y[tgt_idx]
    rmse = np.sqrt(np.mean([(x - y) ** 2 for x, y in zip(true_res, pred_res)]))

    print(r2, r2_new, rmse)
    return r2, r2_new, rmse


def r2_score_os(y_true, y_pred, y_train_mean=0.0):
    assert len(y_true) == len(y_pred)

    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    r2 = np.corrcoef(y_true, y_pred)[0, 1]
    if math.isnan(r2) or r2<0.:
        r2 = 0.
    else:
        r2 = r2 ** 2
    numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y_true - y_train_mean) ** 2).sum(axis=0, dtype=np.float64)
    if denominator == 0:
        return 0., 0.
    assert denominator != 0
    output_scores = 1.0 - (numerator / denominator)
    return r2, output_scores


def testdG(args, epoch, maml, dataloader, is_test=True):
    r2_list_all = []
    rmse_list_all = []
    cir_num = 1
    if is_test:
        cir_num = args.test_repeat_num
    r2_list = []
    r2_new_list = []
    rmse_list = []
    coor_with_similarity = []
    res_dict = {}
    for cir in range(cir_num):
        if is_test:
            test_data_all = dataloader.get_test_batches()
        else:
            test_data_all = dataloader.get_val_batches()
        for step, cur_data in enumerate(test_data_all):
            loss, per_task_target_preds, final_weights, sup_losses, _ = maml.run_validation_iter(cur_data)
            support_loss_each_step = sup_losses[0]
            # loss_pred_zero = sup_losses[0]['loss_pred_zero']

            y = cur_data[1][0].numpy()  # 4:data format, 0:a list, 0:task
            split = cur_data[2][0].numpy()
            y_train = y[np.nonzero(split)]

            if args.datasource == "fsmol":
                std_y_train = y_train.std()
            else:
                std_y_train = max(0.2, y_train.std())
            mean_y_train = y_train.mean()
            if args.qsar:
                if args.datasource == "bdb":
                    bias_y = 6.75021
                elif args.datasource == "drug":
                    bias_y = -2.2329
            else:
                bias_y = 0

            # bias_y = 0
            # std_y_train = 1.
            # mean_y_train = 0.

            y_std = (y - mean_y_train) / std_y_train + bias_y
            
            assay_name = cur_data[3][0]
            if assay_name not in res_dict.keys():
                res_dict[assay_name] = []

            pred_res_std = np.squeeze(per_task_target_preds[0])

            tgt_idx = np.nonzero(1. - split)
            sup_idx = np.nonzero(split)
            sup_y_mean = np.mean(y_std[sup_idx])
            true_res_std = y_std[tgt_idx]
            r2, r2_new = r2_score_os(true_res_std, pred_res_std, sup_y_mean)

            pred_res = (pred_res_std-bias_y)*std_y_train + mean_y_train
            true_res = y[tgt_idx]
            rmse = np.sqrt(np.mean([(x - y) ** 2 for x, y in zip(true_res, pred_res)]))

            res_dict[assay_name].append({
                "r2": r2, "R2os": r2_new, "rmse": rmse,
                "support_loss_each_step": support_loss_each_step
                #, "y": y.tolist(), "split": split.tolist(), "pred_res": pred_res.tolist()
            })
            r2_list.append(r2)
            r2_new_list.append(r2_new)
            rmse_list.append(rmse)


    rmse_i = np.mean(rmse_list)
    rmse_m = np.median(rmse_list)
    res_acc = np.array(r2_list)
    median_r2 = np.median(res_acc, 0)
    mean_r2 = np.mean(res_acc, 0)
    valid_cnt = len([x for x in r2_list if x > 0.3])
    print(
        'epoch is: {}, mean rmse is: {:.3f}, median rmse: {:.3f}'.
        format(epoch, rmse_i, rmse_m))
    print(
        'epoch is: {}, mean is: {:.3f}, median is: {:.3f}, cnt>0.3 is: {:.3f}'.
        format(epoch, mean_r2, median_r2, valid_cnt))
    res_acc = np.array(r2_new_list)
    median_r2os = np.median(res_acc, 0)
    mean_r2os = np.mean(res_acc, 0)
    valid_cnt = len([x for x in r2_new_list if x > 0.3])
    print(
        'epoch is: {}, mean is: {:.3f}, median is: {:.3f}, cnt>0.3 is: {:.3f}'.
        format(epoch, mean_r2os, median_r2os, valid_cnt))
    return res_dict, mean_r2-rmse_i+1


def main():
    maml = MAMLRegressor(args=args, input_shape=(2, args.dim_w))

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
        print("resume training from", model_file)
        try:
            maml.load_state_dict(torch.load(model_file))
        except Exception as e:
            print("*******************************", e)
            maml.load_state_dict(torch.load(model_file), strict=False)

    meta_optimiser = torch.optim.Adam(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)


    if args.datasource == "fsmol":
        dataloader = FsmolMetaLearningSystemDataLoader(args, target_assay=args.target_assay_list,
                                                       exp_string=exp_string)
    elif args.datasource == "gdsc":
        dataloader = GDSCMetaLearningSystemDataLoader(args, target_assay=args.target_assay_list,
                                                       exp_string=exp_string)
    elif args.datasource == "pqsar":
        dataloader = PQSARMetaLearningSystemDataLoader(args, target_assay=args.target_assay_list,
                                                       exp_string=exp_string)
    elif args.datasource in ["ood", "kiba", "covid"]:
        dataloader = EXPERTMetaLearningSystemDataLoader(args, target_assay=args.target_assay_list,
                                                        exp_string=exp_string)
    else:
        dataloader = CHEMBLMetaLearningSystemDataLoader(args, target_assay=args.target_assay_list,
                                                       exp_string=exp_string)

    if args.train == 1:
        train(args, maml, dataloader)
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
                maml.load_state_dict(torch.load(model_file))
            except Exception as e:
                print(e)
                maml.load_state_dict(torch.load(model_file), strict=False)
            res_dict, _ = testdG(args, args.test_epoch, maml, dataloader)
            write_dir = args.test_write_file
            if not os.path.exists(write_dir):
                os.system(f"mkdir -p {write_dir}")
            print("write result to", write_dir, f"sup_num_{args.test_sup_num}.json")
            print("\n\n\n\n")
            json.dump(res_dict, open(os.path.join(write_dir, f"sup_num_{args.test_sup_num}.json"), "w"))
    elif args.train == 4:
        model_file = '{0}/{2}/model_{1}'.format(args.logdir, args.test_epoch, exp_string)
        if not os.path.exists(model_file):
            model_file = '{0}/{1}/model_best'.format(args.logdir, exp_string)
        maml.load_state_dict(torch.load(model_file), strict=False)
        knn_meta_learning(args, maml, dataloader, model_file)

if __name__ == '__main__':
    main()
