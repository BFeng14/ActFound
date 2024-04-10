import random
import torch
from TrainingFramework.ChemUtils import *
#from ProcessControllers import *

class BasicSplitter(object):
    # A splitter module is used to split a dataset
    # with a entire dataset given, the splitter will return the index of the samples of different subsets,
    # or return the subsets directly.
    # return: (sets), (sets_index)
    def __init__(self):
        super(BasicSplitter, self).__init__()

    def split(self, dataset, opt):
        raise NotImplementedError(
            'Dataset splitter not implemented.'
        )

class RandomSplitter(BasicSplitter):
    # Module for randomly splitting dataset
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def CheckClass(self, dataset, tasknum):
        # To check whether both classes of samples appear in the dataset.
        c0cnt = np.zeros(tasknum)
        c1cnt = np.zeros(tasknum)
        for data in dataset:
            value = data['Value']
            assert tasknum == len(value)
            for task in range(tasknum):
                # todo(zqzhang): updated in TPv7
                if (value[task] == '0') or (value[task] == 0):
                    c0cnt[task] += 1
                elif (value[task] == '1') or (value[task] == 1):
                    c1cnt[task] += 1
        if 0 in c0cnt:
            print("Invalid splitting.")
            return False
        elif 0 in c1cnt:
            print("Invalid splitting.")
            return False
        else:
            return True

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        np_dataset = np.array(dataset)
        index = np.arange(total_num)

        if len(rate) == 1:
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
            endflag = 0

            while not endflag:
                random.seed(validseed)
                random.shuffle(index)
                set1_idx = index[:train_num]
                set2_idx = index[train_num:]

                assert len(set1_idx) == train_num
                assert len(set2_idx) == valid_num

                set1 = np_dataset[set1_idx]
                set2 = np_dataset[set2_idx]
                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1
            return (set1, set2), (set1_idx, set2_idx)

        if len(rate) == 2:
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
            endflag = 0

            while not endflag:
                random.seed(testseed)
                random.shuffle(index)
                set3_idx = index[(train_num + valid_num):]
                set3 = np_dataset[set3_idx]

                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set3, opt.args['TaskNum'])
                    testseed += 1
                else:
                    endflag = 1

            set_idx_remain = index[:(train_num + valid_num)]
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(set_idx_remain)

                set1_idx = set_idx_remain[:train_num]
                set2_idx = set_idx_remain[train_num:]
                set1 = np_dataset[set1_idx]
                set2 = np_dataset[set2_idx]

                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1

                assert len(set1) == train_num
                assert len(set2) == valid_num
                assert len(set3) == test_num

            return (set1, set2, set3), (set1_idx, set2_idx, set3_idx)
