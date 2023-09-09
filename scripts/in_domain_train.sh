CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=drug --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_chembl_new/checkpoint_chembl_metricbased   --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --transfer_l &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=drug --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_chembl_new/checkpoint_chembl_ddg_meta      --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg &
CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=drug --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_chembl_new/checkpoint_chembl_qsar_transfer --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar --transfer_l &
CUDA_VISIBLE_DEVICES=3 python main_reg.py --datasource=drug --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_chembl_new/checkpoint_chembl_ddg_transfer  --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg --transfer_l &
CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=drug --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_chembl_new/checkpoint_chembl_qsar_meta     --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar &

CUDA_VISIBLE_DEVICES=0 python main_reg.py --datasource=bdb --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_bdb/checkpoint_bdb_metricbased   --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --transfer_l &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_bdb/checkpoint_bdb_ddg_meta      --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg &
CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=bdb --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_bdb/checkpoint_bdb_qsar_transfer --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar --transfer_l &
CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=bdb --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_bdb/checkpoint_bdb_ddg_transfer  --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg --transfer_l &
CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=bdb --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_bdb/checkpoint_bdb_qsar_meta     --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar &

#CUDA_VISIBLE_DEVICES=7 python main_reg.py --datasource=pqsar --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_pqsar/checkpoint_pqsar_metricbased   --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --transfer_l &
#CUDA_VISIBLE_DEVICES=6 python main_reg.py --datasource=pqsar --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_pqsar/checkpoint_pqsar_ddg_meta      --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg &
#CUDA_VISIBLE_DEVICES=5 python main_reg.py --datasource=pqsar --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_pqsar/checkpoint_pqsar_qsar_transfer --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar --transfer_l &
#CUDA_VISIBLE_DEVICES=4 python main_reg.py --datasource=pqsar --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_pqsar/checkpoint_pqsar_ddg_transfer  --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --new_ddg --transfer_l &
#CUDA_VISIBLE_DEVICES=1 python main_reg.py --datasource=pqsar --metatrain_iterations=50 --update_lr=0.001 --meta_lr=0.00015 --min_learning_rate 0.0001 --num_updates=5 --test_num_updates=5 --trial=1 --meta_batch_size=16 --drug_group=1 --logdir ./checkpoints_pqsar/checkpoint_pqsar_qsar_meta     --hid_dim 2048 --sim_thres 0.2 --dim_w 2048 --qsar &
