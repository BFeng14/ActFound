FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="../meta_delta/checkpoints_chembl"
CHEMBL_RES="./result_indomain/chembl"
#CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_meta --model_name meta_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} ${CHEMBL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_transfer --model_name transfer_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_meta --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_transfer --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="../meta_delta/checkpoints_bdb"
BDB_RES="./result_indomain/bdb"
#CUDA_VISIBLE_DEVICES=2 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_meta --model_name meta_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_transfer --model_name transfer_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_meta --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_transfer --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM} &

FIXED_PARAM_PQSAR="--test_repeat_num 1 --train 0 --test_epoch -1"
PQSAR_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/pqsar/feat.npy --train_assay_idxes ./train_assay_feat/pqsar/index.pkl"
PQSAR_DIR="../meta_delta/checkpoints_pqsar"
PQSAR_RES="./result_indomain/pqsar"
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_ddg_meta --model_name meta_delta --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} ${PQSAR_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_ddg_transfer --model_name transfer_delta --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_qsar_meta --model_name maml --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_protonet --model_name protonet --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_qsar_transfer --model_name transfer_qsar --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &

FIXED_PARAM_FSMOL="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="../meta_delta/checkpoints_fsmol"
FSMOL_RES="./result_indomain/fsmol"
#CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_meta --model_name meta_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_transfer --model_name transfer_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_meta --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_transfer --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 32 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="../meta_delta/checkpoints_fsmol"
FSMOL_RES="./result_indomain/fsmol"
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_meta --model_name meta_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_transfer --model_name transfer_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_meta --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_transfer --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 64 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="../meta_delta/checkpoints_fsmol"
FSMOL_RES="./result_indomain/fsmol"
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_meta --model_name meta_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_transfer --model_name transfer_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_meta --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_transfer --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 128 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="../meta_delta/checkpoints_fsmol"
FSMOL_RES="./result_indomain/fsmol"
CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_meta --model_name meta_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_ddg_transfer --model_name transfer_delta --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_meta --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
#CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=fsmol --logdir ${FSMOL_DIR}/checkpoint_fsmol_qsar_transfer --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &