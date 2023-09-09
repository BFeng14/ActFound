FIXED_PARAM="--metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --train 0 --test_epoch -1 --test_sup_num 16 --expert_test ood"
SAVE_DIR="./test_result_ood_ac_minus"
CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=drug ${FIXED_PARAM} --logdir ./checkpoints_chembl/checkpoint_chembl_metricbased   --transfer_l           --test_write_file "${SAVE_DIR}/chembl/protonet" &
CUDA_VISIBLE_DEVICES=2 python main_reg.py --datasource=drug ${FIXED_PARAM} --logdir ./checkpoints_chembl/checkpoint_chembl_ddg_meta_nonorm      --new_ddg       --test_write_file "${SAVE_DIR}/chembl/Meta-DDG-nonorm"  --transfer_lr 0.008 &
CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=drug ${FIXED_PARAM} --logdir ./checkpoints_chembl/checkpoint_chembl_qsar_transfer --qsar --transfer_l    --test_write_file "${SAVE_DIR}/chembl/Transfer-DG"  --transfer_lr 0.002 &
CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=drug ${FIXED_PARAM} --logdir ./checkpoints_chembl/checkpoint_chembl_ddg_transfer  --new_ddg --transfer_l --test_write_file "${SAVE_DIR}/chembl/Transfer-DDG" &
CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=drug ${FIXED_PARAM} --logdir ./checkpoints_chembl/checkpoint_chembl_qsar_meta     --qsar                 --test_write_file "${SAVE_DIR}/chembl/Meta-DG" &

CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb ${FIXED_PARAM} --logdir ./checkpoints_bdb/checkpoint_bdb_metricbased   --transfer_l            --test_write_file "${SAVE_DIR}/bdb/protonet" &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb ${FIXED_PARAM} --logdir ./checkpoints_bdb/checkpoint_bdb_ddg_meta_nonorm  --new_ddg            --test_write_file "${SAVE_DIR}/bdb/Meta-DDG-nonorm" &
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=bdb ${FIXED_PARAM} --logdir ./checkpoints_bdb/checkpoint_bdb_qsar_transfer --qsar --transfer_l     --test_write_file "${SAVE_DIR}/bdb/Transfer-DG" --transfer_lr 0.004 &
CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=bdb ${FIXED_PARAM} --logdir ./checkpoints_bdb/checkpoint_bdb_ddg_transfer  --new_ddg --transfer_l  --test_write_file "${SAVE_DIR}/bdb/Transfer-DDG" --transfer_lr 0.004 &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=bdb ${FIXED_PARAM} --logdir ./checkpoints_bdb/checkpoint_bdb_qsar_meta     --qsar                  --test_write_file "${SAVE_DIR}/bdb/Meta-DG" &
