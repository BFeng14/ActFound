from ACComponents.ACProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SubsetsNum': 3,
    'OnlyEval': False,
}

BasicParamList = {
    'ExpName': 'ACLarge',
    'MainMetric': 'AUC',
    'DataPath': './ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Large.json',
    'RootPath': './TestExp/Large/LSTM/',
    'CUDA_VISIBLE_DEVICES': '0',
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,
    'Feature': 'SMILES',
    'Model': 'LSTM',

    'OnlySpecific': True,
    'Weight': True,
    'AC': True,
    'PyG': False,

    'ValidRate': 40000,
    'PrintRate': 5,
    'UpdateRate': 1,
    'SplitRate': [0.8, 0.1],
    'Splitter': 'Random',
    'MaxEpoch': 300,
    'LowerThanMaxLimit': 12,
    'DecreasingLimit': 8,

    # if OnlyEval == True:
    'EvalModelPath': None,
    'EvalDatasetPath': None,
    'EvalLogAllPreds': None,

    'Scheduler': 'PolynomialDecayLR',

    # Params for PolynomialDecayLR only
    'WarmupEpoch': 2,
    'LRMaxEpoch':300,
    'EndLR':1e-9,
    'Power':1.0,
    # Params for StepLR only
    'LRStep': 30,
    'LRGamma': 0.1,
    ##########

    'WeightIniter': None,

    # Params for NormWeightIniter only
    'InitMean' : 0,
    'InitStd' : 1,

    'AtomFeatureSize': 39,
    'BondFeatureSize': 10,
    'MolFP': 'MorganFP',
    'radius': 2,
    'nBits': 1024,

    'SplitValidSeed': 8,
    'SplitTestSeed': 8,
    'BatchSize': 256,

}
AdjustableParamList = {}
SpecificParamList = {
    'DropRate':[0.4],
    'WeightDecay':[5],
    'lr':[3],
    'LSTMLayers': [3],
    'FPSize':[512],
    'DNNLayers':[[512, 128, 32]],
}


expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

