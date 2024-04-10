from TrainingFramework.Splitter import *


class TargetSplitter(BasicSplitter):
    def __init__(self):
        super(TargetSplitter, self).__init__()

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        if total_num == 1:
            dataset = dataset[0]
            total_num = len(dataset)

        tarid2size, tarid2sample = self.GetTargetidList(dataset)
        tarids = tarid2size.keys()

        # calculate the splitting thres
        if len(rate) == 1:
            assert rate[0] < 1
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
        elif len(rate) == 2:
            assert rate[0] + rate[1] < 1
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
        else:
            print("Wrong splitting rate")
            raise RuntimeError

        if len(rate) == 1:
            sample_size = int(len(tarids) * (1-rate[0]))
            validtargets, chosen_cnt = self.BinaryClassSample(tarid2size, tarids, sample_size, valid_num, validseed)
            validset, valididx = self.Target2Samples(validtargets,tarid2sample)
            assert len(validset) == chosen_cnt
            traintargets = self.excludedtargets(validtargets, tarids)
            trainset, trainidx = self.Target2Samples(traintargets, tarid2sample)
            assert len(validset) + len(trainset) == total_num
            return (trainset, validset), (trainidx, valididx)
        elif len(rate) == 2:
            sample_size = int(len(tarids) * (1-rate[0]-rate[1]))
            testtargets, chosen_cnt = self.BinaryClassSample(tarid2size, tarids, sample_size, test_num, testseed)
            testset, testidx = self.Target2Samples(testtargets, tarid2sample)
            assert len(testset) == chosen_cnt
            remained_tarids = self.excludedtargets(testtargets, tarids)
            sample_size = int(len(tarids) * rate[1])
            validtargets, chosen_cnt = self.BinaryClassSample(tarid2size, remained_tarids, sample_size, valid_num, validseed)
            validset, valididx = self.Target2Samples(validtargets, tarid2sample)
            assert len(validset) == chosen_cnt
            traintargets = self.excludedtargets(validtargets, remained_tarids)
            trainset, trainidx = self.Target2Samples(traintargets, tarid2sample)
            assert len(validset)+len(testset)+len(trainset) == total_num
            return (trainset, validset, testset), (trainidx, valididx, testidx)

    def BinaryClassSample(self, tarid2size, tarids, sample_size, optimal_count, seed):

        count = 0
        tried_times = 0
        error_rate = 0.1


        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)):
            tried_times += 1

            if tried_times % 5000 == 0:
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample target number.")
                sample_size = int(sample_size * 1.1)
                assert sample_size < len(tarids)

            seed += 1
            random.seed(seed)
            chosen_targets = random.sample(tarids, sample_size)
            count = sum([tarid2size[target] for target in chosen_targets])

        print(f"Sample num: {count}")
        print(f"Tried times: {tried_times}")
        print(f"Available seed: {seed}")


        return chosen_targets, count

    def Target2Samples(self, chosen_targets, tarid2sample):
        set = []
        for targetid in chosen_targets:
            targetset = tarid2sample[targetid]
            set.extend(targetset)
        idx = []
        for item in set:
            id = item['idx']
            idx.append(id)
        return set, idx

    def excludedtargets(self, chosen_targets, tarids):
        excluded_targets = []
        for target in tarids:
            if target not in chosen_targets:
                excluded_targets.append(target)
        return excluded_targets

    def GetTargetidList(self, dataset):
        tarid2size = {}
        tarid2sample = {}
        for item in dataset:
            tarid = item['Target']
            if tarid not in tarid2size.keys():
                tarid2size.update({tarid: 0})
                tarid2sample.update({tarid: []})
            tarid2size[tarid] += 1
            tarid2sample[tarid].append(item)
        return tarid2size, tarid2sample


def verification(sets, opt):
    rate = opt.args['SplitRate']
    if len(rate)==1:
        trainset, validset = sets
        testset = None
    elif len(rate) == 2:
        trainset, validset, testset = sets

    train_targets = []
    valid_targets = []
    test_targets = []

    for item in trainset:
        target = item['Target']
        if target not in train_targets:
            train_targets.append(target)

    for item in validset:
        target = item['Target']
        if target not in valid_targets:
            valid_targets.append(target)

    if testset:
        for item in testset:
            target = item['Target']
            if target not in test_targets:
                test_targets.append(target)

    # varify train and valid
    for target in train_targets:
        assert target not in valid_targets

    for target in valid_targets:
        assert target not in train_targets

    # verify train and test
    if testset:
        for target in train_targets:
            assert target not in test_targets
        for target in test_targets:
            assert target not in train_targets

        # verify valid and test
        for target in valid_targets:
            assert target not in test_targets
        for target in test_targets:
            assert target not in valid_targets

    print(f"Verification passed.")
    print(f"trainset target num: {len(train_targets)}")
    print(f"validset target num: {len(valid_targets)}")
    print(f"testset target num: {len(test_targets)}")
