FIXED_PARAM_DAVIS="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_DAVIS="./test_results/result_cross/chembl2davis"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES_DAVIS="./test_results/result_cross/bdb2davis"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_maml --model_name maml --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_transfer_qsar --model_name transfer_qsar --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
FIXED_PARAM_KIBA="--test_sup_num [64,128] --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test kiba"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_KIBA="./test_results/result_cross/chembl2kiba"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES_KIBA="./test_results/result_cross/bdb2kiba"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_maml --model_name maml --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_transfer_qsar --model_name transfer_qsar --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
