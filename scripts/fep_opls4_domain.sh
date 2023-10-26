FIXED_PARAM="--test_sup_num [0.2,0.4,0.6,0.8] --test_repeat_num 40 --train 0 --test_epoch -1 --expert_test fep_opls4"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_fep_new/fep_opls4/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
#
BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_fep_new/fep_opls4/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_maml --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_transfer_qsar --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM} &
