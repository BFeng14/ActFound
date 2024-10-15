# ActFound


## General
This repository contains the code for ActFound (now published on Nature Machine Intelligence): A bioactivity foundation model using pairwise meta-learning.

- Link to this paper: https://www.nature.com/articles/s42256-024-00876-w
- Free share link: https://rdcu.be/dQUav

## Colab programmatic tool
⭐ We have now added a ready-to-use online programmatic tool on Colab.

😊 You can easily use Colab tools to fine-tune Actfound with few measured compounds and then use it to give prediction on your unmeasured compounds.

❤️ We also provide a metric in this colab to help you predict if Actfound can works well on your data with only few measured compounds for fine-tuning. (See Figure 3.g.h of our paper for more detailes)

- Colab for Actfound

    - https://colab.research.google.com/drive/1zn3U3xwLXZjZQYGVgtxCiEwt1hkho62l?usp=sharing 

- Colab for Actfound with KNN-MAML and fusion method

    - https://colab.research.google.com/drive/1eLWidAOWUqSCEcm0qM0Pf4IO1Ex7Ceal?usp=sharing

- Colab for finetuning full parameters of Actfound (can be more suitable when you have many measured compounds for finetuning, e.g. more than 100)

    - https://colab.research.google.com/drive/1EPx2tMHIdvhPbY8GyvuICVwVgwd3ikfJ?usp=sharing

## Abstract
Compound bioactivity plays an important role in drug development and discovery. Existing machine learning approaches have poor generalization ability in compound bioactivity prediction due to the small number of compounds in each assay and incompatible measurements among assays. In this paper, we propose ActFound, a bioactivity foundation model trained on 1.6 million experimentally measured bioactivities and 35,644 assays from ChEMBL. The key idea of ActFound is to employ pairwise learning to learn the relative value differences between two compounds within the same assay to circumvent the incompatibility among assays. ActFound further exploits meta-learning to jointly optimize the model from all assays. On six real-world bioactivity datasets, ActFound demonstrates accurate in-domain prediction and strong generalization across datasets, assay types, and molecular scaffolds. We also demonstrated that ActFound can be used as an accurate alternative to the leading physics-based computational tool FEP+(OPLS4) by achieving comparable performance when only using a few data points for fine-tuning. The promising results of ActFound indicate that ActFound can be an effective bioactivity foundation model for a wide range of tasks in compound bioactivity prediction, paving the path for machine learning-based drug development and discovery.
## Instructions for runing Actfound

### Download data
All data, model checkpoints, and test results are available on the Google Drive.
- Download Link
    - Google Drive: https://drive.google.com/drive/folders/1x-F_hbQr_pXFEA5qLCkd7dIr9a_1L3aJ?usp=drive_link
    - Figshare: https://figshare.com/articles/dataset/ActFound_data/24452680
- Please first download all files necessary, and unzip them all. Please put "datas" dir under the project root, so that the code can find it during running.
- For model inference, please download checkpoints_all.tar.gz, extract all files, and put them in the project root.
- To use models trained with strict data leakage processing method (all compounds in FEP benchmarks are excluded and correlation ($\rho_P$) threshold of data leakage is 0.95), please download checkpoints_bdb_nofep.tar.gz and checkpoints_chembl_nofep.tar.gz from google drive, extract them all and put the extracted dir into "checkpoints_all".
- For plot figures in our paper, please download test_results_all.tar.gz, extract all files, put them in the project root, and rename the dir into test_results.
- Please make sure that "datas", "checkpoints_all"(necessary for inference), "test_results"(necessary for plot) all correctly placed in to project root.

### Reproduce the results in our paper
#### In-domain bioactivity prediction
Experiments on ChEMBL and BindingDB datasets:
- Note that KNN_MAML is optional for testing, and it will first take about half an hour to run few-shot testing on all assays on ChEMBL(only needed once), and then take another half hour to fully run on the test set of ChEMBL. You can just close it by not setting --knn_maml. 
- To get a better result, please run another inference using ActFound(transfer), and fusion the result of ActFound and ActFound(transfer) using "fusion_result.py". The result of our paper utilized both KNN_MAML and the fusion method.
```bash
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"

CHEMBL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/chembl/feat.npy --train_assay_idxes ./train_assay_feat/chembl/index.pkl"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_indomain/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} ${CHEMBL_KNN_MAML} &

BDB_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/bdb/feat.npy --train_assay_idxes ./train_assay_feat/bdb/index.pkl"
BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_indomain/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
```
+ **'model_name'**: Specify the method to run, choose between actfound, actfound_transfer, maml, protonet and transfer_qsar
+ **'test_sup_num'**: Specify the size of support set

Experiments on pQSAR-ChEMBL:
```bash
FIXED_PARAM_PQSAR="--test_sup_num 0.75 --test_repeat_num 1 --train 0 --test_epoch -1"
PQSAR_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/pqsar/feat.npy --train_assay_idxes ./train_assay_feat/pqsar/index.pkl"
PQSAR_DIR="./checkpoints_all/checkpoints_pqsar"
PQSAR_RES="./test_results/result_indomain/pqsar"
python main_reg.py --datasource=pqsar --logdir ${PQSAR_DIR}/checkpoint_pqsar_actfound --model_name actfound --test_write_file ${PQSAR_RES} ${FIXED_PARAM_PQSAR} ${PQSAR_KNN_MAML} &
```
+ **'model_name'**: Specify the method to run, choose between actfound, actfound_transfer, maml, protonet and transfer_qsar
+ **'test_sup_num'**: Specify the percentage used for fine-tuning

Experiments on FS-MOL:
```bash
FIXED_PARAM_FSMOL="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1"
FSMOL_KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/fsmol/feat.npy --train_assay_idxes ./train_assay_feat/fsmol/index.pkl"
FSMOL_DIR="./checkpoints_all/checkpoints_fsmol"
FSMOL_RES="./test_results/result_indomain/fsmol"
python main_reg.py --datasource=kiba --logdir ${FSMOL_DIR}/checkpoint_fsmol_actfound --model_name actfound --test_write_file ${FSMOL_RES} ${FIXED_PARAM_FSMOL} ${FSMOL_KNN_MAML} &
```
+ **'model_name'**: Specify the method to run, choose among actfound, actfound_transfer, maml, protonet and transfer_qsar
+ **'test_sup_num'**: Specify the size of support set, choose among 16, 32, 64, 128

Experiments on ChEMBL-Activity:
```bash
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test ood"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_ood"
CHEMBL_RES_TMP="./test_results/result_ood_tmp"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file "${CHEMBL_RES_TMP}/normal" ${FIXED_PARAM}
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file "${CHEMBL_RES_TMP}/inverse" ${FIXED_PARAM} --inverse_ylabel
python ./combine_inverse_prediction.py
```
#### Cross-domain bioactivity prediction
ChEMBL to BindingDB:
```bash
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --cross_test"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_cross/chembl2bdb"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &
```

BindingDB to ChEMBL:
```bash
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --cross_test"

BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_cross/bdb2chembl"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} &
```

ChEMBL to DAVIS and BindingDB to DAVIS:
```bash
FIXED_PARAM_DAVIS="--test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test davis"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_DAVIS="./test_results/result_cross/chembl2davis"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES_DAVIS} ${FIXED_PARAM_DAVIS} &

BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES_DAVIS="./test_results/result_cross/bdb2davis"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES_DAVIS} ${FIXED_PARAM_DAVIS} &
```

ChEMBL to DAVIS and BindingDB to DAVIS:
```bash
FIXED_PARAM_KIBA="--test_sup_num [16,32,64,128] --test_repeat_num 10 --train 0 --test_epoch -1 --expert_test kiba"
CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES_KIBA="./test_results/result_cross/chembl2kiba"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES_KIBA} ${FIXED_PARAM_KIBA} &

BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES_KIBA="./test_results/result_cross/bdb2kiba"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES_KIBA} ${FIXED_PARAM_KIBA} &
```
#### FEP prediction
Experiment bioactivity as the fine-tuning set:
```bash
FIXED_PARAM="--test_sup_num [0.2,0.4,0.6,0.8] --test_repeat_num 40 --train 0 --test_epoch -1 --expert_test fep"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_fep_new/fep/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_fep_new/fep/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
```
FEP+(OPLS4) calculation results as the fine-tuning set:
```bash
FIXED_PARAM="--test_sup_num [0.2,0.4,0.6,0.8] --test_repeat_num 40 --train 0 --test_epoch -1 --expert_test fep_opls4"

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl"
CHEMBL_RES="./test_results/result_fep_new/fep_opls4/chembl"
python main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} &

BDB_DIR="./checkpoints_all/checkpoints_bdb"
BDB_RES="./test_results/result_fep_new/fep_opls4/bdb"
python main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} ${BDB_KNN_MAML} &
```
#### Cancer drug response prediction
```bash
FIXED_PARAM="--model_name actfound --test_write_file test_results/gdsc --test_repeat_num 1 --metatrain_iterations 80 --input_celline"
python main_reg.py --datasource=gdsc ${FIXED_PARAM} --logdir ./checkpoints_all/checkpoints_chembl/checkpoint_chembl_actfound --resume 1 --test_epoch -1  --gdsc_pretrain_data chembl
python main_reg.py --datasource=gdsc ${FIXED_PARAM} --logdir ./checkpoints_all/checkpoints_bdb/checkpoint_bdb_actfound --resume 1 --test_epoch -1  --gdsc_pretrain_data bdb
python main_reg.py --datasource=gdsc ${FIXED_PARAM}
```

We provide all test results on the file test_results_all.tar.gz, which can be used to plot all figures shown in our paper.
we also provide all scripts we used to run the full test in "./script". You can refer to the correct running script contained in "./script" 

### Train ActFound yourself
#### Run model training
For ActFound training, please first make sure that the training data is correctly downloaded, and then simply run "main_reg.py". Training of ActFound on ChEMBL takes roughly 70 hours, and training on BindingDB takes roughly 30 hours.
- For training on other datasets, please replace DATA_SOURCE. 
- For training on other models, please refer to "learning_system/\_\_init__.py", and change the MODEL_NAME.

```bash
FIXED_PARAM="--test_sup_num 16 --test_repeat_num 2 --begin_lrloss_epoch 50 --metatrain_iterations 80 --no_fep_lig "

CHEMBL_DIR="./checkpoints_all/checkpoints_chembl_nofep"
CHEMBL_RES="./test_results/result_indomain_nofep/chembl"
python -u main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound --model_name actfound --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} > ./runlog_nofep/chembl_actfound.log &
python -u main_reg.py --datasource=chembl --logdir ${CHEMBL_DIR}/checkpoint_chembl_actfound_transfer --model_name actfound_transfer --test_write_file ${CHEMBL_RES} ${FIXED_PARAM} > ./runlog_nofep_new/chembl_actfound_transfer.log &


BDB_DIR="./checkpoints_all/checkpoints_bdb_nofep"
BDB_RES="./test_results/result_indomain_nofep/bdb"
python -u main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound --model_name actfound --test_write_file ${BDB_RES} ${FIXED_PARAM} > ./runlog_nofep/bdb_actfound.log &
python -u main_reg.py --datasource=bdb --logdir ${BDB_DIR}/checkpoint_bdb_actfound_transfer --model_name actfound_transfer --test_write_file ${BDB_RES} ${FIXED_PARAM} > ./runlog_nofep_new/bdb_actfound_transfer.log &
```

Please feel free to contact me by email if there is any problem with the code or paper: fengbin14@pku.edu.cn.

## Citation
If you use ActFound in your work, a citation to our paper is appreciated:
- DOI: https://doi.org/10.1038/s42256-024-00876-w
- Link to this paper: https://www.nature.com/articles/s42256-024-00876-w
- Free link of NMI version: https://rdcu.be/dQUav
