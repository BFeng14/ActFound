import numpy as np
from torch.utils import data
from TrainingFramework.Featurizer import *
from TrainingFramework.FileUtils import *
from TrainingFramework.Splitter import *
from ACComponents.ACChemUtils import *
from ACComponents.ACSplitter import *
from TrainingFramework.Dataset import PyGMolDataset
from Models.Graphormer.collator import collator

class ACDataset(data.Dataset):
    def __init__(self, dataset, opt, mode):
        super(ACDataset, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.BuildFeaturizer()
        self.decompose_pairwise_dataset()

        if (opt.args['Feature'] == 'SMILES'):
            self.featurizer1.prefeaturize(self.decomposed_dataset1)
            self.featurizer2.prefeaturize(self.decomposed_dataset2)

    def BuildFeaturizer(self):
        feature_choice = self.opt.args['Feature']
        if feature_choice == 'FP':
            self.featurizer1 = FPFeaturizer(self.opt)
            self.featurizer2 = FPFeaturizer(self.opt)
        elif feature_choice == 'Graph':
            self.featurizer1 = GraphFeaturizer()
            self.featurizer2 = GraphFeaturizer()
        elif feature_choice == 'CMPNN':
            self.featurizer1 = CMPNNFeaturizer(self.opt)
            self.featurizer2 = CMPNNFeaturizer(self.opt)
        elif feature_choice == 'SMILES':
            self.featurizer1 = SMILESTokenFeaturizer(self.opt)
            self.featurizer2 = SMILESTokenFeaturizer(self.opt)
        elif feature_choice == 'Graphormer':
            self.featurizer1 = GraphormerFeaturizer(
                    max_node = self.opt.args['max_node'],
                    multi_hop_max_dist = self.opt.args['multi_hop_max_dist'],
                    spatial_pos_max = self.opt.args['spatial_pos_max'],
                    opt = self.opt
            )
            self.featurizer2 = GraphormerFeaturizer(
                    max_node = self.opt.args['max_node'],
                    multi_hop_max_dist = self.opt.args['multi_hop_max_dist'],
                    spatial_pos_max = self.opt.args['spatial_pos_max'],
                    opt = self.opt
            )
        elif feature_choice == 'Raw':
            self.featurizer1 = PretrainFeatureFeaturizer(self.opt)
            self.featurizer2 = PretrainFeatureFeaturizer(self.opt)
        else:
            raise KeyError("Wrong feature option!")

    def decompose_pairwise_dataset(self):
        self.decomposed_dataset1 = []
        self.decomposed_dataset2 = []
        if not self.opt.args['Finetune']:
            for item in self.dataset:
                smiles1 = item['SMILES1']
                smiles2 = item['SMILES2']
                value = item['Value']
                self.decomposed_dataset1.append({'SMILES': smiles1, 'Value': value})
                self.decomposed_dataset2.append({'SMILES': smiles2, 'Value': value})
        else:
            for item in self.dataset:
                feature1 = item['Feature1']
                feature2 = item['Feature2']
                value = item['Value']
                self.decomposed_dataset1.append({'Feature': feature1, 'Value': value})
                self.decomposed_dataset2.append({'Feature': feature2, 'Value': value})
        assert len(self.decomposed_dataset1) == len(self.decomposed_dataset2)


    def __getitem__(self, index):
        if (self.featurizer1.__class__ == GraphormerFeaturizer):
            item1 = self.decomposed_dataset1[index]
            item2 = self.decomposed_dataset2[index]
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            data1 = self.featurizer1.featurize(item1)
            data2 = self.featurizer2.featurize(item2)
            data1.append(idx)
            data2.append(idx)
            return (data1, data2)
        else:
            item1 = self.decomposed_dataset1[index]
            item2 = self.decomposed_dataset2[index]
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            data1, label = self.featurizer1.featurize(item1)
            data2, label = self.featurizer2.featurize(item2)
        return (data1, data2), label, idx

    def __len__(self):
        return len(self.dataset)

class ACMolDatasetCreator(object):
    # An object to create molecule datasets from a given dataset file path.
    # Using CreateDatasets function to generate 2 or 3 datasets, based on the SplitRate
    # Based on the MolDatasetCreator above, this version added the MSN creating part
    # including the network building and the mask creating according to the splitting.
    def __init__(self, opt, subsetindex):
        super(ACMolDatasetCreator, self).__init__()
        self.SplitterList = {
            'Random': RandomSplitter(),
            'TargetRandom': TargetSplitter()
        }
        self.opt = opt
        self.subsetindex = subsetindex

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == '0':
                    neg_count += 1
                elif value == '1':
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self):
        file_path = self.opt.args['DataPath']
        print("Loading data file...")
        fileloader = JsonFileLoader(file_path)
        raw_data = fileloader.load()

        print("Extracting AC subsets")
        keys = raw_data.keys()
        keys_list = []
        for iii, key in enumerate(keys):
            keys_list.append(key)
        key = keys_list[self.subsetindex]
        raw_dataset = raw_data[key]

        print("Dataset is parsed. Original size is ", len(raw_dataset))

        # raw_dataset is the original dataset from the files
        # before processing, checker should be used to screen the samples that not satisfy the rules.
        # if using AttentiveFP models, extra rules are used to check the dataset according to the Attentive FP.
        # otherwise, only MolChecker is used to make sure that all of the samples are validity to the rdkit.


        if  self.opt.args['Feature'] == 'Graphormer':
            print('Using checking rules proposed in Attentive FP. Dataset is being checked.')
            self.checker = ACAttentiveFPChecker(max_atom_num=102, max_degree=5, pair_wise = True)
            self.screened_dataset = self.checker.check(raw_dataset)
        else:
            self.checker = ACMolChecker(pair_wise=True)
            self.screened_dataset = self.checker.check(raw_dataset)

        for idx, data in enumerate(self.screened_dataset):
            data.update({'idx': idx})

        self.CheckScreenedDatasetIdx()

        # After checking, all of the "raw_dataset" should be taken place by "self.screened_dataset"
        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None
        # weights is a list with length of 'TaskNum'.
        # It shows the distribution of pos/neg samples in the dataset(Screened dataset, before splitting and after screening)
        # And it is used as a parameter of the loss function to balance the learning process.
        # For multitasks, it contains multiple weights.

        if self.opt.args['Splitter']:
            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets, idxs = splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)


        # Construct dataset objects of subsets
        if not self.opt.args['PyG']:
            Trainset = ACDataset(trainset, self.opt, 'TRAIN')
            if len(sets) == 2:
                Validset = ACDataset(validset, self.opt, 'EVAL')
                return (Trainset, Validset), weights
            elif len(sets) == 3:
                Validset = ACDataset(validset, self.opt, 'EVAL')
                Testset = ACDataset(testset, self.opt, 'EVAL')
                return (Trainset, Validset, Testset), weights
            else:
                return (Trainset), weights
        else:
            pyg_feater = PyGACGraphFeaturizer(self.opt)
            PyGGraphTrainset = []
            for sample in trainset:
                graph_sample = pyg_feater.featurize(sample)
                PyGGraphTrainset.append(graph_sample)

            Trainset = PyGMolDataset(PyGGraphTrainset, self.opt, 'TRAIN')

            if len(sets) == 2:
                PyGGraphValidset = []
                for sample in validset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphValidset.append(graph_sample)
                Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                return (Trainset, Validset), weights
            elif len(sets) == 3:
                PyGGraphValidset = []
                for sample in validset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphValidset.append(graph_sample)
                Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                PyGGraphTestset = []
                for sample in testset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphTestset.append(graph_sample)
                Testset = PyGMolDataset(PyGGraphTestset, self.opt, 'TEST')
                return (Trainset, Validset, Testset), weights



    def CheckScreenedDatasetIdx(self):
        print("Check whether idx is correct: ")
        chosen_idx = int(random.random() * len(self.screened_dataset))
        print(chosen_idx)
        print(self.screened_dataset[chosen_idx])
        assert chosen_idx == self.screened_dataset[chosen_idx]['idx']


class ACPTMDatasetCreator(object):
    # An object to create molecule datasets from a given dataset file path.
    # Using CreateDatasets function to generate 2 or 3 datasets, based on the SplitRate
    # Based on the MolDatasetCreator above, this version added the MSN creating part
    # including the network building and the mask creating according to the splitting.
    def __init__(self, opt, subsetindex):
        super(ACPTMDatasetCreator, self).__init__()
        self.SplitterList = {
            'Random': RandomSplitter(),
            'TargetRandom': TargetSplitter()
        }
        self.opt = opt
        self.subsetindex = subsetindex

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                # print(f"item: {item}")
                value = item['Value'][i]
                if value == 0:
                    neg_count += 1
                elif value == 1:
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self):
        file_path = self.opt.args['DataPath']
        print("Loading data file...")
        fileloader = NpzFileLoader(file_path)
        raw_data = fileloader.load()

        print("Extracting AC subsets")
        Features = raw_data[raw_data.files[self.subsetindex*2]]
        Labels = raw_data[raw_data.files[self.subsetindex*2+1]]
        assert len(Features) == len(Labels)
        print(f"The size of the subset is: {len(Features)}")

        raw_dataset = []
        for i in range(len(Features)):
            Feature = Features[i]
            Label = Labels[i]
            if Label.__class__ == np.ndarray:
                Label = int(Label.item())
            assert len(Feature) == 2
            item = {'Feature1': Feature[0], 'Feature2': Feature[1], 'Value': [Label]}
            raw_dataset.append(item)



        print("Dataset is parsed. Original size is ", len(raw_dataset))

        # Checker is not need for PTM extracted features
        self.screened_dataset = raw_dataset

        # raw_dataset is the original dataset from the files
        # before processing, checker should be used to screen the samples that not satisfy the rules.
        # if using AttentiveFP models, extra rules are used to check the dataset according to the Attentive FP.
        # otherwise, only MolChecker is used to make sure that all of the samples are validity to the rdkit.


        for idx, data in enumerate(self.screened_dataset):
            data.update({'idx': idx})

        self.CheckScreenedDatasetIdx()

        # After checking, all of the "raw_dataset" should be taken place by "self.screened_dataset"
        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None
        # weights is a list with length of 'TaskNum'.
        # It shows the distribution of pos/neg samples in the dataset(Screened dataset, before splitting and after screening)
        # And it is used as a parameter of the loss function to balance the learning process.
        # For multitasks, it contains multiple weights.

        if self.opt.args['Splitter']:
            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets, idxs = splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)


        # Construct dataset objects of subsets
        if not self.opt.args['PyG']:
            Trainset = ACDataset(trainset, self.opt, 'TRAIN')
            if len(sets) == 2:
                Validset = ACDataset(validset, self.opt, 'EVAL')
                return (Trainset, Validset), weights
            elif len(sets) == 3:
                Validset = ACDataset(validset, self.opt, 'EVAL')
                Testset = ACDataset(testset, self.opt, 'EVAL')
                return (Trainset, Validset, Testset), weights
            else:
                return (Trainset), weights
        else:
            pyg_feater = PyGACGraphFeaturizer(self.opt)
            PyGGraphTrainset = []
            for sample in trainset:
                graph_sample = pyg_feater.featurize(sample)
                PyGGraphTrainset.append(graph_sample)
            # print(PyGGraphTrainset)
            # print(len(PyGGraphTrainset))
            Trainset = PyGMolDataset(PyGGraphTrainset, self.opt, 'TRAIN')
            # print(Trainset)
            # print(len(Trainset))
            if len(sets) == 2:
                PyGGraphValidset = []
                for sample in validset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphValidset.append(graph_sample)
                Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                return (Trainset, Validset), weights
            elif len(sets) == 3:
                PyGGraphValidset = []
                for sample in validset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphValidset.append(graph_sample)
                Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                PyGGraphTestset = []
                for sample in testset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphTestset.append(graph_sample)
                Testset = PyGMolDataset(PyGGraphTestset, self.opt, 'TEST')
                return (Trainset, Validset, Testset), weights



    def CheckScreenedDatasetIdx(self):
        print("Check whether idx is correct: ")
        chosen_idx = int(random.random() * len(self.screened_dataset))
        print(chosen_idx)
        print(self.screened_dataset[chosen_idx])
        assert chosen_idx == self.screened_dataset[chosen_idx]['idx']


def ACcollator(items, max_node, multi_hop_max_dist, spatial_pos_max):
    batch_size = len(items)
    items1 = []
    items2 = []
    for sample in items:
        items1.append(sample[0])
        items2.append(sample[1])
    assert len(items1) == len(items2) == batch_size

    Batch1 = collator(items1, max_node, multi_hop_max_dist, spatial_pos_max)
    Batch2 = collator(items2, max_node, multi_hop_max_dist, spatial_pos_max)

    return (Batch1, Batch2)




