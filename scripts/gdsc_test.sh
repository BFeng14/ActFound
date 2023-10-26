FIXED_PARAM="--model_name actfound --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 80 --input_celline"
python main_reg.py --datasource=gdsc ${FIXED_PARAM} --logdir ./checkpoints_all/checkpoints_chembl/checkpoint_chembl_actfound --resume 1 --test_epoch -1  --gdsc_pretrain_data chembl
python main_reg.py --datasource=gdsc ${FIXED_PARAM} --logdir ./checkpoints_all/checkpoints_bdb/checkpoint_bdb_actfound --resume 1 --test_epoch -1  --gdsc_pretrain_data bdb
python main_reg.py --datasource=gdsc ${FIXED_PARAM}