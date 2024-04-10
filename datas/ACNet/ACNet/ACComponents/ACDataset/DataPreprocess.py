import pandas as pd
import os
import numpy as np
from ACComponents.ACDataset.DataUtils import Config, SaveJson, LoadJson
import random

OriginDatasetAddrAll = './data_files/raw_data/all_smiles_target.csv'
OriginDatasetAddrPos = './data_files/raw_data/mmp_ac_s_distinct.csv'
OriginDatasetAddrNeg = './data_files/raw_data/mmp_ac_s_neg_distinct.csv'

GeneratedDatasetAddrAll = './data_files/generated_datasets/MMP_AC.json'
GeneratedDatasetAddrLarge = './data_files/generated_datasets/MMP_AC_Large.json'
GeneratedDatasetAddrMedium = './data_files/generated_datasets/MMP_AC_Medium.json'
GeneratedDatasetAddrSmall = './data_files/generated_datasets/MMP_AC_Small.json'
GeneratedDatasetAddrFew = './data_files/generated_datasets/MMP_AC_Few.json'

DiscardedDatasetAddr = './data_files/generated_datasets/MMP_AC_Discarded.json'

GeneratedDatasetAddrMixed = './data_files/generated_datasets/MMP_AC_Mixed.json'
GeneratedDatasetAddrMixedScreened = './data_files/generated_datasets/MMP_AC_Mixed_Screened.json'



def ReadACDatafile(AddrPos, AddrNeg):
    dfpos = pd.read_csv(AddrPos)
    dfneg = pd.read_csv(AddrNeg)

    targets = {}

    total_items1 = len(dfpos)
    total_items2 = len(dfneg)

    for i in range(total_items1):
        target = str(dfpos['tid'][i])
        if target not in targets.keys():
            targets.update({target:[]})

        SMILES1 = dfpos['c1'][i]
        SMILES2 = dfpos['c2'][i]
        targets[target].append({'SMILES1': SMILES1, 'SMILES2': SMILES2, 'Value': '1'})

    discard_cnt = 0
    valid_neg_cnt = 0
    for i in range(total_items2):
        target = str(dfneg['tid'][i])
        if target in targets.keys():
            SMILES1 = dfneg['c1'][i]
            SMILES2 = dfneg['c2'][i]
            targets[target].append({'SMILES1': SMILES1, 'SMILES2': SMILES2, 'Value': '0'})
            valid_neg_cnt += 1
        if target not in targets.keys():
            discard_cnt += 1

    print(f"Total positive count: {total_items1}")
    print(f"Total negative count: {total_items2}")
    print(f"Valid negative count: {valid_neg_cnt}")
    print(f"Discarded negative count: {discard_cnt}")

    return targets

def RandomScreenNeg(dataset, config):
    screened_dataset = {}
    discarded_dataset = {}

    org_tot_cnt = 0
    allowed_ratio = config.pn_rate_threshold
    for target in dataset.keys():
        print(f"Checking tid:{target}")
        subset = dataset[target]
        org_tot_cnt += len(subset)
        pos_set = []
        neg_set = []
        for item in subset:
            if item['Value'] == '1':
                pos_set.append(item)
            else:
                neg_set.append(item)
        pos_cnt = len(pos_set)
        neg_cnt = len(neg_set)
        print(f"Pos/Neg ratio: {pos_cnt/neg_cnt}.")

        if (pos_cnt / neg_cnt) > allowed_ratio:
            print(f"Allowed.")
            screened_dataset.update({target:subset})
        else:
            print(f"Screening...")
            screened_subset = pos_set.copy()
            max_sample_num = int(pos_cnt / allowed_ratio)
            print(f"Pos cnt: {pos_cnt}.")
            random.seed(config.random_sample_negative_seed)
            random.shuffle(neg_set)
            chosen_neg = neg_set[:max_sample_num]
            print(f"Randomly chosen: {len(chosen_neg)}")
            discarded_subset = neg_set[max_sample_num:]
            print(f"Discard: {len(discarded_subset)}")
            screened_subset.extend(chosen_neg)
            print(f"subset after screening: {len(screened_subset)}")
            print(f"ratio after screening: {len(pos_set) / len(chosen_neg)}.")
            screened_dataset.update({target:screened_subset})
            discarded_dataset.update({target:discarded_subset})

    print(f"Dataset after screening: {len(screened_dataset)}")
    tot_cnt = 0
    dis_cnt = 0
    for key in screened_dataset.keys():
        subset = screened_dataset[key]
        tot_cnt+=len(subset)
    for key in discarded_dataset.keys():
        subset = discarded_dataset[key]
        dis_cnt+=len(subset)
    print(f"Number of samples reserved:{tot_cnt}")
    print(f"Number of samples discarded:{dis_cnt}")
    assert (tot_cnt + dis_cnt) == org_tot_cnt

    return screened_dataset, discarded_dataset








def SubsetNumDistribution(dataset):
    cnt = 0
    cnt_distribution = []
    cnt_distribution_tid = []
    for idx, tid in enumerate(dataset):
        item = dataset[(str(tid))]
        cnt_distribution_tid.append(tid)
        cnt_distribution.append(len(item))
        cnt += len(item)

    print(f"Total number of samples in the dataset: {cnt}")
    print(f"Size of all Subsets in the dataset: {cnt_distribution}")
    print(f"The tid of all subsets in the dataset: {cnt_distribution_tid}")
    print(f"The maximum size of subsets: {max(cnt_distribution)}")
    print(f"The minimum size of subsets: {min(cnt_distribution)}")

    return cnt, cnt_distribution, cnt_distribution_tid

def SplitSubsetsByCnt(dataset, cnt_distribution, cnt_distribution_tid, config):
    cnt_distribution = np.array(cnt_distribution)

    large_thres = config.large_thres
    medium_thres = config.medium_thres
    small_thres = config.small_thres

    cnt_subset_large = np.where(cnt_distribution > large_thres)[0]
    cnt_subset_medium = np.where((cnt_distribution <= large_thres)&(cnt_distribution > medium_thres))[0]
    cnt_subset_small = np.where((cnt_distribution <= medium_thres)&(cnt_distribution > small_thres))[0]
    cnt_subset_few = np.where((cnt_distribution <= small_thres)&(cnt_distribution > 1))[0]

    print(f"The number of subsets in Large set is: {len(cnt_subset_large)}")
    print(f"The number of subsets in Medium set is: {len(cnt_subset_medium)}")
    print(f"The number of subsets in Small set is: {len(cnt_subset_small)}")
    print(f"The number of subsets in Few set is: {len(cnt_subset_few)}")

    subset_large = {}
    subset_medium = {}
    subset_small = {}
    subset_few = {}

    for i in range(len(cnt_subset_large)):
        loc = cnt_subset_large[i]
        tid = cnt_distribution_tid[loc]
        item = dataset[tid]
        subset_large.update({tid: item})
    SaveJson(GeneratedDatasetAddrLarge, subset_large)


    for i in range(len(cnt_subset_medium)):
        loc = cnt_subset_medium[i]
        tid = cnt_distribution_tid[loc]
        item = dataset[tid]
        subset_medium.update({tid: item})
    SaveJson(GeneratedDatasetAddrMedium, subset_medium)

    for i in range(len(cnt_subset_small)):
        loc = cnt_subset_small[i]
        tid = cnt_distribution_tid[loc]
        item = dataset[tid]
        subset_small.update({tid: item})
    SaveJson(GeneratedDatasetAddrSmall, subset_small)

    for i in range(len(cnt_subset_few)):
        loc = cnt_subset_few[i]
        tid = cnt_distribution_tid[loc]
        item = dataset[tid]
        subset_few.update({tid: item})
    SaveJson(GeneratedDatasetAddrFew, subset_few)

def ScreenFewPosSubsets(dataset, config):
    screened_dataset = dataset.copy()
    discarded_subsets = {}
    few_pos_threshold = config.few_pos_threshold

    for idx, tid in enumerate(dataset):
        print(f"Checking subset of target {tid}")
        item = dataset[str(tid)]
        subset_num = len(item)
        print(f"Total num of samples of this target: {subset_num}")
        pos_cnt = 0
        for i in range(subset_num):
            sample = item[i]
            if sample['Value'] == '1':
                pos_cnt += 1

        print(f"Total positive sample num {pos_cnt}")
        if pos_cnt < few_pos_threshold:
            print(f"Discard this subset.")
            screened_dataset.pop(str(tid))
            discarded_subsets.update({str(tid): item})

    assert len(screened_dataset) + len(discarded_subsets) == len(dataset)
    return screened_dataset, discarded_subsets

def ScreenImbalancedSubsets(dataset, config):
    screened_dataset = dataset.copy()
    discarded_subsets = {}
    pn_rate_threshold = config.pn_rate_threshold

    for idx, tid in enumerate(dataset):
        print(f"Checking subset of target {tid}")
        item = dataset[str(tid)]
        subset_size = len(item)
        print(f"Total num of samples of this target: {subset_size}")

        pos_cnt = 0
        neg_cnt = 0
        for i in range(subset_size):
            sample = item[i]
            if sample['Value'] == '1':
                pos_cnt += 1
            elif sample['Value'] == '0':
                neg_cnt += 1
            else:
                raise ValueError(
                        f'Wrong Value of target {tid} and sample {sample} with idx {i}.'
                )

        rate = pos_cnt / neg_cnt
        print(f"Positive / Negative rate is: {rate}")

        if rate < pn_rate_threshold:
            print(f"Discard this subset.")
            screened_dataset.pop(str(tid))
            discarded_subsets.update({str(tid): item})

    assert len(screened_dataset) + len(discarded_subsets) == len(dataset)
    return screened_dataset, discarded_subsets

def MixAllSubsets(dataset):
    mixed_dataset = {'All':[]}

    # total_targets_num = len(mixed_dataset)
    for idx, tid in enumerate(dataset):
        item = dataset[str(tid)]
        subset_size = len(item)

        for i in range(subset_size):
            sample = item[i]
            sample.update({'Target': tid})
            mixed_dataset['All'].append(sample)

    return mixed_dataset

def CheckConflictSamples(dataset):
    total_num = len(dataset)
    print(f"Total number of samples in mixed dataset is {total_num}.")

    MolPairDict = {}
    conflict_cnt = 0
    for item in dataset:
        smiles1 = item['SMILES1']
        smiles2 = item['SMILES2']
        molpair = smiles1 + '?' + smiles2
        molpair_rev = smiles2 + '?' + smiles1
        if (molpair not in MolPairDict.keys()) & (molpair_rev not in MolPairDict.keys()):
            MolPairDict.update({molpair: item})
        else:
            if molpair in MolPairDict.keys():
                conflict_molpair = molpair
            elif molpair_rev in MolPairDict.keys():
                conflict_molpair = molpair_rev
            previous_value = MolPairDict[conflict_molpair]['Value']
            current_value = item['Value']
            if previous_value != current_value:
                print(f"Confilict encountered!")
                conflict_cnt += 1
                print(f"Previous conflict sample: {conflict_molpair} : {MolPairDict[conflict_molpair]}.")
                print(f"Current sample: {item}.")
    print(f"Total conflict sample number is {conflict_cnt}")

def ScreenConflictSamples(dataset):
    total_num = len(dataset)
    print(f"Total number of samples in the mixed dataset is {total_num}.")

    MolPairDict = {}
    ScreenedDataset = []
    MolPairIndexDict = {}
    ToBeScreenedMolPairList = []
    discarded_cnt = 0
    repeated_cnt = 0
    for item in dataset:
        smiles1 = item['SMILES1']
        smiles2 = item['SMILES2']
        molpair = smiles1 + '?' + smiles2
        molpair_rev = smiles2 + '?' + smiles1

        if (molpair not in MolPairDict.keys()) & (molpair_rev not in MolPairDict.keys()):
            MolPairDict.update({molpair: item})
            ScreenedDataset.append(item)
            MolPairIndexDict.update({molpair: len(ScreenedDataset)})

        else:
            repeated_cnt += 1
            if molpair in MolPairDict.keys():
                conflict_molpair = molpair
            elif molpair_rev in MolPairDict.keys():
                conflict_molpair = molpair_rev
            previous_value = MolPairDict[conflict_molpair]['Value']
            current_value = item['Value']

            if previous_value != current_value:
                # previous_index = MolPairIndexDict[conflict_molpair]
                # previous_item = ScreenedDataset[previous_index]
                # previous_molpair = previous_item['SMILES1'] + '?' + previous_item['SMILES2']
                # assert previous_molpair == conflict_molpair
                # ScreenedDataset.pop(previous_index)
                if MolPairDict[conflict_molpair] not in ToBeScreenedMolPairList:
                    ToBeScreenedMolPairList.append(MolPairDict[conflict_molpair])
                discarded_cnt += 1

    for item in ToBeScreenedMolPairList:
        try:
            ScreenedDataset.remove(item)
        except:
            print(f"{item} have been removed before.")

    print(f"Total repeated sample number is {repeated_cnt}.")
    print(f"Total discarded sample number is {discarded_cnt}.")
    print(f"Total to be screened sample number is {len(ToBeScreenedMolPairList)}.")
    print(f"Remained sample number is {len(ScreenedDataset)}.")

    return ScreenedDataset


####################################


def ACDatasetPreprocess(config):
    if not os.path.exists(GeneratedDatasetAddrAll):
        dataset = ReadACDatafile(OriginDatasetAddrPos, OriginDatasetAddrNeg)
        discarded_dataset = {}
        if config.discard_few_pos:
            screened_dataset, discarded_dataset1 = ScreenFewPosSubsets(dataset, config)
            discarded_dataset.update(discarded_dataset1)
        if config.random_sample_negative:
            screened_dataset, discarded_dataset2 = RandomScreenNeg(screened_dataset, config)
            discarded_dataset.update(discarded_dataset2)
        if config.discard_extreme_imbalance:
            screened_dataset, discarded_dataset3 = ScreenImbalancedSubsets(screened_dataset, config)
            discarded_dataset.update(discarded_dataset2)

        SaveJson(GeneratedDatasetAddrAll, screened_dataset)
        SaveJson(DiscardedDatasetAddr, discarded_dataset)

        dataset = screened_dataset

    else:
        dataset = LoadJson(GeneratedDatasetAddrAll)

    print(f'Total targets(subsets) of the dataset: {len(dataset)}')

    cnt, cnt_distribution, cnt_distribution_tid = SubsetNumDistribution(dataset)

    SplitSubsetsByCnt(dataset, cnt_distribution, cnt_distribution_tid, config)

    if config.mixed:
        mixed_dataset = MixAllSubsets(dataset)
        print(f"Total number of samples in the mixed dataset is {len(mixed_dataset['All'])}")
        SaveJson(GeneratedDatasetAddrMixed, mixed_dataset)
        CheckConflictSamples(mixed_dataset['All'])
        screened_mixed_dataset = {'All': []}
        screened_mixed_dataset['All'] = ScreenConflictSamples(mixed_dataset['All'])
        SaveJson(GeneratedDatasetAddrMixedScreened, screened_mixed_dataset)



