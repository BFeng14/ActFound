FIXED_PARAM="--test_sup_num [0.2,0.3,0.4,0.5,0.6,0.7,0.8] --test_repeat_num 40 --train 0 --test_epoch -1 --expert_test fep_opls4"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="../meta_delta/checkpoints_chembl"
CHEMBL_RES="./result_fep/fep_opls4/chembl"
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_meta --model_name meta_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} ${CHEMBL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_transfer --model_name transfer_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_meta --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_transfer --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="../meta_delta/checkpoints_bdb"
BDB_RES="./result_fep/fep_opls4/bdb"
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_meta --model_name meta_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_transfer --model_name transfer_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_meta --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_transfer --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM} &
