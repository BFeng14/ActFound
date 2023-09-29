FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test ood"

CHEMBL_DIR="../meta_delta/checkpoints_chembl"
CHEMBL_RES="./result_ood/chembl"
CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_meta --model_name meta_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_ddg_transfer --model_name transfer_delta --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_meta --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_qsar_transfer --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_DIR="../meta_delta/checkpoints_bdb"
BDB_RES="./result_ood/bdb"
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_meta --model_name meta_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_ddg_transfer --model_name transfer_delta --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_meta --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM} &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_qsar_transfer --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM} &
