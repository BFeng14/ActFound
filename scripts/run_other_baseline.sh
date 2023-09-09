CHEMBL_TEST_PATH="/home/fengbin/datas/split_in_domain_chembl.json"
CHEMBL_TEST_PATH_CROSS="/home/fengbin/datas/split_cross_domain_bdb2chembl.json"
CHEMBL_FEAT_PATH="/home/fengbin/datas/chembl/CHEMBL_FEAT_ALL"

BDB_TEST_PATH="/home/fengbin/datas/split_in_domain_bdb.json"
BDB_TEST_PATH_CROSS="/home/fengbin/datas/split_cross_domain_chembl2bdb.json"
BDB_FEAT_PATH="/home/fengbin/datas/BDB/BDB_FEAT_ALL"

KIBA_TEST_PATH="/home/fengbin/datas/expert/kiba_split/split_all_test.json"
KIBA_FEAT_PATH="/home/fengbin/datas/expert/kiba_feat"

DAVIS_TEST_PATH="/home/fengbin/datas/expert/davis_split/split_all_test.json"
DAVIS_FEAT_PATH="/home/fengbin/datas/expert/davis_feat"

ACT_TEST_PATH="/home/fengbin/datas/chembl/ood_activity.json"
ACT_FEAT_PATH="/home/fengbin/datas/chembl/CHEMBL_FEAT_Activity"

CNP_CHEMBL_CKPT="/mnt/sfs_turbo/fengbin/ADKF-IFT/checkpoint_chembl_new/CNP_model"
DKT_CHEMBL_CKPT="/mnt/sfs_turbo/fengbin/ADKF-IFT/outputs/FSMol_DKTModel_ecfp+fc_2023-09-03_23-15-36"
CNP_BDB_CKPT="/mnt/sfs_turbo/fengbin/ADKF-IFT/outputs/FSMol_CNPModel_ecfp+fc_2023-09-03_23-15-23"
DKT_BDB_CKPT="/mnt/sfs_turbo/fengbin/ADKF-IFT/outputs/FSMol_DKTModel_ecfp+fc_2023-09-04_10-28-17"

RP="/home/fengbin/meta_delta"
# in-domain
cd /mnt/sfs_turbo/fengbin/ADKF-IFT
#CUDA_VISIBLE_DEVICES=0 python fs_mol/cnp_test.py "${CNP_CHEMBL_CKPT}/best_validation.pt" "${CHEMBL_FEAT_PATH}" \
#       --task-list-file ${CHEMBL_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result/chembl/CNP" &
#CUDA_VISIBLE_DEVICES=1 python fs_mol/dkt_test.py "${DKT_CHEMBL_CKPT}/best_validation.pt" "${CHEMBL_FEAT_PATH}" \
#       --task-list-file ${CHEMBL_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result/chembl/DKT" &
#
#CUDA_VISIBLE_DEVICES=2 python fs_mol/cnp_test.py "${CNP_BDB_CKPT}/best_validation.pt" "${BDB_FEAT_PATH}" \
#       --task-list-file ${BDB_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result/bdb/CNP" &
CUDA_VISIBLE_DEVICES=3 python fs_mol/dkt_test.py "${DKT_BDB_CKPT}/best_validation.pt" "${BDB_FEAT_PATH}" \
       --task-list-file ${BDB_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result/bdb/DKT" &
#
#
## cross-domain
#CUDA_VISIBLE_DEVICES=4 python fs_mol/cnp_test.py "${CNP_CHEMBL_CKPT}/best_validation.pt" "${BDB_FEAT_PATH}" \
#       --task-list-file ${BDB_TEST_PATH_CROSS} --train-sizes "[16]" --save-dir "${RP}/test_result_cross/chembl2bdb/CNP" &
#CUDA_VISIBLE_DEVICES=5 python fs_mol/dkt_test.py "${DKT_CHEMBL_CKPT}/best_validation.pt" "${BDB_FEAT_PATH}" \
#       --task-list-file ${BDB_TEST_PATH_CROSS} --train-sizes "[16]" --save-dir "${RP}/test_result_cross/chembl2bdb/DKT" &
#
#CUDA_VISIBLE_DEVICES=6 python fs_mol/cnp_test.py "${CNP_BDB_CKPT}/best_validation.pt" "${CHEMBL_FEAT_PATH}" \
#       --task-list-file ${CHEMBL_TEST_PATH_CROSS} --train-sizes "[16]" --save-dir "${RP}/test_result_cross/bdb2chembl/CNP" &
CUDA_VISIBLE_DEVICES=7 python fs_mol/dkt_test.py "${DKT_BDB_CKPT}/best_validation.pt" "${CHEMBL_FEAT_PATH}" \
       --task-list-file ${CHEMBL_TEST_PATH_CROSS} --train-sizes "[16]" --save-dir "${RP}/test_result_cross/bdb2chembl/DKT"


## kiba domain
#CUDA_VISIBLE_DEVICES=0 python fs_mol/cnp_test.py "${CNP_CHEMBL_CKPT}/best_validation.pt" "${KIBA_FEAT_PATH}" \
#       --task-list-file ${KIBA_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_kiba/chembl/CNP" &
#CUDA_VISIBLE_DEVICES=1 python fs_mol/dkt_test.py "${DKT_CHEMBL_CKPT}/best_validation.pt" "${KIBA_FEAT_PATH}" \
#       --task-list-file ${KIBA_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_kiba/chembl/DKT" &
#
#CUDA_VISIBLE_DEVICES=2 python fs_mol/cnp_test.py "${CNP_BDB_CKPT}/best_validation.pt" "${KIBA_FEAT_PATH}" \
#       --task-list-file ${KIBA_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_kiba/bdb/CNP" &
CUDA_VISIBLE_DEVICES=3 python fs_mol/dkt_test.py "${DKT_BDB_CKPT}/best_validation.pt" "${KIBA_FEAT_PATH}" \
       --task-list-file ${KIBA_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_kiba/bdb/DKT" &

## davis domain
#CUDA_VISIBLE_DEVICES=4 python fs_mol/cnp_test.py "${CNP_CHEMBL_CKPT}/best_validation.pt" "${DAVIS_FEAT_PATH}" \
#       --task-list-file ${DAVIS_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_davis/chembl/CNP" &
#CUDA_VISIBLE_DEVICES=5 python fs_mol/dkt_test.py "${DKT_CHEMBL_CKPT}/best_validation.pt" "${DAVIS_FEAT_PATH}" \
#       --task-list-file ${DAVIS_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_davis/chembl/DKT" &
#
#CUDA_VISIBLE_DEVICES=6 python fs_mol/cnp_test.py "${CNP_BDB_CKPT}/best_validation.pt" "${DAVIS_FEAT_PATH}" \
#       --task-list-file ${DAVIS_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_davis/bdb/CNP" &
CUDA_VISIBLE_DEVICES=7 python fs_mol/dkt_test.py "${DKT_BDB_CKPT}/best_validation.pt" "${DAVIS_FEAT_PATH}" \
       --task-list-file ${DAVIS_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_davis/bdb/DKT"
# activity domain
#CUDA_VISIBLE_DEVICES=4 python fs_mol/cnp_test.py "${CNP_CHEMBL_CKPT}/best_validation.pt" "${ACT_FEAT_PATH}" \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/chembl/CNP" &
#CUDA_VISIBLE_DEVICES=5 python fs_mol/dkt_test.py "${DKT_CHEMBL_CKPT}/best_validation.pt" "${ACT_FEAT_PATH}" \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/chembl/DKT" &
#
#CUDA_VISIBLE_DEVICES=6 python fs_mol/cnp_test.py "${CNP_BDB_CKPT}/best_validation.pt" "${ACT_FEAT_PATH}" \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/bdb/CNP" &
CUDA_VISIBLE_DEVICES=7 python fs_mol/dkt_test.py "${DKT_BDB_CKPT}/best_validation.pt" "${ACT_FEAT_PATH}" \
       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/bdb/DKT" &


#python fs_mol/baseline_numeric_test_knn.py "${ACT_FEAT_PATH}" \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/KNN" &
#python fs_mol/baseline_numeric_test.py     "${ACT_FEAT_PATH}" \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/RF" &
#python fs_mol/gpst_test.py                 "${ACT_FEAT_PATH}" --use-numeric-labels \
#       --task-list-file ${ACT_TEST_PATH} --train-sizes "[16]" --save-dir "${RP}/test_result_ood_ac/GPST" &
