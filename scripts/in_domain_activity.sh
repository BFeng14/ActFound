FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test ood"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_ood"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_maml --model_name maml --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_protonet --model_name protonet --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_transfer_qsar --model_name transfer_qsar --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
CHEMBL_RES_TMP="./test_results/result_ood_tmp"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file "${CHEMBL_RES_TMP}/normal" ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file "${CHEMBL_RES_TMP}/inverse" ${FIXED_PARAM} --inverse_ylabel
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file "${CHEMBL_RES_TMP}/normal" ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file "${CHEMBL_RES_TMP}/inverse" ${FIXED_PARAM} --inverse_ylabel
python ./combine_inverse_prediction.py