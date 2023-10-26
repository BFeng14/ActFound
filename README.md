# ActFound


## General
This repository contains the code for ActFound: A foundation model for bioactivity prediction using pairwise meta-learning.

## Abstract
Compound bioactivity plays an important role in drug development and discovery. Existing machine learning approaches have poor generalization ability due to the small number of compounds in each assay and incompatible measurements among assays. Here, we propose ActFound, a foundation model for bioactivity value prediction trained X assays and Y compounds from ChEMBL and BindingDB. The key idea of ActFound is to employ pairwise learning to learn the relative value differences between two compounds within the same assay to circumvent the incompatibility among assays. ActFound further exploits meta-learning to jointly optimize the model from all assays. On four real-world bioactivity datasets, ActFound demonstrates accurate in-domain prediction and strong generalization across datasets, units, and molecular scaffolds. We also demonstrated that ActFound can be used as an accurate alternative to the leading computational chemistry software FEP+(OPLS4) by achieving comparable performance when only using a few data points for fine-tuning. The promising results of ActFound indicate that ActFound can be an effective foundation model for a wide range of tasks in compound bioactivity prediction, paving the path for machine learning-based drug development and discovery.

## Instructions for runing the code

### download data
All datas, model checkpoints and test results are available on the Google Drive.
- Download Link: https://drive.google.com/drive/folders/1x-F_hbQr_pXFEA5qLCkd7dIr9a_1L3aJ?usp=drive_link
- Please first download all files necessary, and unzip them all. Please put "datas" dir under the project root, so that the code can find it during running.
- For model inference, please download checkpoints_all.tar.gz, extract all files and put them the project root.
- For plot figure in our paper, please download test_results_all.tar.gz, extract all files, put them the project root, and rename the dir into test_results.
- Please make sure that "datas", "checkpoints_all"(necessary for inference), "test_results"(necessary for plot) all correctly placed in to project root.

### run model inference
For model inference only, please download the corresponding file from the google drive, and simply run "main_reg.py". The checkpoints for models trained on ChEMBL, BindingDB, FS-MOL and pQSAR-ChEMBL are avaliable on the google drive. 
- Note that KNN_MAML is optional for testing, and it will first take about half hour to run few-shot testing on all assays on ChEMBL(only needed once), and then take another half hour to fully run on the test set of ChEMBL. You can just close it by not setting --knn_maml. one example for runing in-domain test on ChEMBL is given below.
- For testing following in-domain setting, please just simply replace the DATA_SCOURCE (e.g. BindingDB), and it's corresponding MODEL_DIR.
- For testing on BindingDB using model trained on ChEMBL, please simple add --cross_test.
- For testing on other dataset (KIBA, Davis, FEP, fep_opls4, activity) using model trained on ChEMBL, please simple add "--expert_test test_domain_name". (fep_opls4 means that using the result of FEP+(OPLS4) for fine-tuning, and activity means ChEMBL-Activity)
- For testing using other models (including ActFound(transfer), MAML, ProtoNet, TransferQSAR), please refer to "learning_system/\_\_init__.py", and change the MODEL_NAME.
- To get better result, please run another inference using ActFound(transfer), and fusion the result of ActFound and ActFound(transfer) using "fusion_result.py". The result on our paper utilzed both KNN_MAML and fusion method.
- For inference on other datasources, please follow the code in "dataset/load_dataset.py" (for example read_FEP_SET), and load your dataset in "dataset/data_chemblbdb_Assay_reg.py".
```bash
DATA_SOURCE="chembl"
MODEL_DIR="path_to_load_model_checkpoint"
MODEL_NAME="actfound"
RESULT_FILE="path_to_result_file"
KNN_MAML="--knn_maml --train_assay_feat_all ./train_assay_feat/${DATA_SOURCE}/feat.npy --train_assay_idxes ./train_assay_feat/${DATA_SOURCE}/index.pkl"
python main_reg.py --datasource=${DATA_SOURCE} --logdir ${MODEL_DIR} --model_name ${MODEL_NAME} --test_write_file ${RESULT_FILE} --test_sup_num 16 --test_repeat_num 10 --train 0 --test_epoch -1 ${KNN_MAML}
```

### run model training
For ActFound training, please first make sure that the training data is correctly downloaded, and then simply run "main_reg.py". Training of ActFound on ChEMBL takes nearly 3 days, training on BindingDB takes 30 hours.
- For training on other dataset, please replace DATA_SOURCE. 
- For training on other model, please refer to "learning_system/\_\_init__.py", and change the MODEL_NAME.

```bash
DATA_SOURCE="chembl"
MODEL_DIR="path_to_save_model_checkpoint"
MODEL_NAME="actfound"
python main_reg.py --datasource=${DATA_SOURCE} --logdir ${MODEL_DIR} --model_name ${MODEL_NAME} --test_write_file ${RESULT_FILE} --test_sup_num 16 --test_repeat_num 2 
```

### result on paper reproduce
if you want to reproduce the result shown in our paper, we provide all scripts we used to run the full test in "./script". You can refer to the correct running script is contained in "./script" if you find are not sure about how to run inference.

## Citing
If you use MetaLigand in your work, a citation to our paper is appreciated:
