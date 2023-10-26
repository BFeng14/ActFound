FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_indomain/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} ${CHEMBL_KNN_MAML} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_indomain/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_maml --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_transfer_qsar --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM} &

FIXED_PARAM_PQSAR="--test_repeat_num 1 --train 0 --test_epoch -1"
PQSAR_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/pqsar/feat.npy --train_assay_idxes ./train_assay_feat/pqsar/index.pkl"
PQSAR_DIR="./checkpoints_all/checkpoints_pqsar"
PQSAR_RES="./test_results/result_indomain/pqsar"
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_actfound --model_name actfound --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} ${PQSAR_KNN_MAML} &
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_actfound_transfer --model_name actfound_transfer --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_maml --model_name maml --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_protonet --model_name protonet --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_transfer_qsar --model_name transfer_qsar --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} &

FIXED_PARAM_FSMOL="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
CUDA_VISIBLE_DzVICES=7 python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound_transfer --model_name actfound_transfer --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_maml --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_transfer_qsar --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 32 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound_transfer --model_name actfound_transfer --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_maml --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_transfer_qsar --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 64 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound_transfer --model_name actfound_transfer --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_maml --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_transfer_qsar --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &

FIXED_PARAM_FSMOL="--test_sup_num 128 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound_transfer --model_name actfound_transfer --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_maml --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_transfer_qsar --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} &