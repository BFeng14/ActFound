FIXED_PARAM_DAVIS="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="../meta_delta/checkpoints_chembl"
CHEMBL_RES_DAVIS="./result_cross/chembl2davis"
#CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_meta --model_name meta_delta --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_transfer --model_name transfer_delta --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_meta --model_name maml --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_transfer --model_name transfer_qsar --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="../meta_delta/checkpoints_bdb"
BDB_RES_DAVIS="./result_cross/bdb2davis"
#CUDA_VISIBLE_DEVICES=2 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_meta --model_name meta_delta --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_transfer --model_name transfer_delta --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=2 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_meta --model_name maml --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_transfer --model_name transfer_qsar --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
FIXED_PARAM_KIBA="--test_sup_num [16,32,64,128] --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test kiba"
CHEMBL_DIR="../meta_delta/checkpoints_chembl"
CHEMBL_RES_KIBA="./result_cross/chembl2kiba"
CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_meta --model_name meta_delta --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_transfer --model_name transfer_delta --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_meta --model_name maml --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_transfer --model_name transfer_qsar --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="../meta_delta/checkpoints_bdb"
BDB_RES_KIBA="./result_cross/bdb2kiba"
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_meta --model_name meta_delta --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_transfer --model_name transfer_delta --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_meta --model_name maml --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_transfer --model_name transfer_qsar --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
