FIXED_PARAM="--test_sup_num 16 --test_repeat_num 2"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl --begin_lrloss_epoch 50 --metatrain_iterations 60"
CHEMBL_RES="./test_results/result_indomain/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES} ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM}

BDB_DIR="./checkpoints_all/checkpoints_bdb --begin_lrloss_epoch 50 --metatrain_iterations 60"
BDB_RES="./test_results/result_indomain/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM}
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES} ${FIXED_PARAM}
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_maml --model_name maml --test_write_file ${BDB_RES} ${FIXED_PARAM}
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_protonet --model_name protonet --test_write_file ${BDB_RES} ${FIXED_PARAM}
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_transfer_qsar --model_name transfer_qsar --test_write_file ${BDB_RES} ${FIXED_PARAM}

FIXED_PARAM_PQSAR="--test_repeat_num 1 --begin_lrloss_epoch 60 --metatrain_iterations 80"
PQSAR_DIR="./checkpoints_all/checkpoints_pqsar"
PQSAR_RES="./test_results/result_indomain/pqsar"
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_actfound --model_name actfound --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR}
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_actfound_transfer --model_name actfound_transfer --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR}
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_maml --model_name maml --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR}
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_protonet --model_name protonet --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR}
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_transfer_qsar --model_name transfer_qsar --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR}

FIXED_PARAM_FSMOL="--test_sup_num 16 --test_repeat_num 10 --begin_lrloss_epoch 60 --metatrain_iterations 80"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL}
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound_transfer --model_name actfound_transfer --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL}
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_maml --model_name maml --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL}
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_protonet --model_name protonet --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL}
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_transfer_qsar --model_name transfer_qsar --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL}
