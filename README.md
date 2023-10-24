# meta_delta

## General
This repository contains the code accompanying the paper

## Abstract

## Instructions for runing the code

### download data


### run model inference
For model inference only, please download the corresponding file from the google drive, and simply run main_reg.py. The checkpoints for models trained on ChEMBL, BindingDB, FS-MOL and pQSAR-ChEMBL are avaliable on the google drive. 
- Note that KNN_MAML is optional for testing, and it will first take about half hour to run few-shot testing on all assays on ChEMBL(only needed once), and then take another half hour to fully run on the test set of ChEMBL. You can just close it by not setting --knn_maml. one example for runing in-domain test on ChEMBL is given below.
- For testing following in-domain setting, please just simply replace the DATA_SCOURCE (e.g. BindingDB), and it's corresponding MODEL_DIR.
- For testing on BindingDB using model trained on ChEMBL, please simple add --cross_test.
- For testing on other dataset (KIBA, Davis, FEP, fep_opls4, activity) using model trained on ChEMBL, please simple add "--expert_test test_domain_name". (fep_opls4 means that using the result of FEP+(OPLS4) for fine-tuning, and activity means ChEMBL-Activity)
- For testing using other models (including ActFound(transfer), MAML, ProtoNet, TransferQSAR), please refer to learning_system/__init__.py, and change the MODEL_NAME.
- To get better result, please run another inference using ActFound(transfer), and fusion the result of ActFound and ActFound(transfer) using fusion_result.py. The result on our paper utilzed both KNN_MAML and fusion method.

```bash
DATA_SCOURCE="chembl"
MODEL_DIR="path_to_load_model_checkpoint"
MODEL_NAME="actfound"
RESULT_FILE="path_to_result_file"
KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/${DATA_SCOURCE}/feat.npy --train_assay_idxes ./train_assay_feat/${DATA_SCOURCE}/index.pkl"
python main_reg.py --datasource=${DATA_SCOURCE} --logdir ${MODEL_DIR} --model_name ${MODEL_NAME} --test_write_file ${RESULT_FILE} --test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 ${KNN_MAML}
```

### run model training
For ActFound training, please first make sure that the training data is correctly downloaded, and then simply run main_reg.py. Training of ActFound on ChEMBL takes nearly 3 days, training on BindingDB takes 30 hours.
- For training on other dataset, please replace DATA_SCOURCE. 
- For training on other model, please refer to learning_system/__init__.py, and change the MODEL_NAME.

```bash
DATA_SCOURCE="chembl"
MODEL_DIR="path_to_save_model_checkpoint"
MODEL_NAME="actfound"
python main_reg.py --datasource=${DATA_SCOURCE} --logdir ${MODEL_DIR} --model_name ${MODEL_NAME} --test_write_file ${RESULT_FILE} --test_sup_num 16 --test_repeat_num 2 
```

### result on paper reproduce
if you want to reproduce the result shown in our paper, we provide all scripts we used to run the full test in ./script.

## Citing
If you use MetaLigand in your work, a citation to our paper is appreciated:
