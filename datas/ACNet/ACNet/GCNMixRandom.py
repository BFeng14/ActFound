from ACComponents.ACProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SubsetsNum': 1,
    'OnlyEval': False,
}

BasicParamList = {
    'ExpName': 'ACMixRandom',
    'MainMetric': 'AUC',
    'DataPath': './ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Mixed_Screened.json',
    'RootPath': './TestExp/Mix/GCN/',
    'CUDA_VISIBLE_DEVICES': '1',
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,
    'Feature': 'PyGGCN',
    'Model': 'PyGGCN',

    'OnlySpecific': True,
    'Weight': True,
    'AC': True,
    'PyG': True,

    'ValidRate': 4000,
    'PrintRate': 5,
    'UpdateRate': 1,
    'SplitRate': [0.8, 0.1],
    'Splitter': 'Random',
    'MaxEpoch': 300,
    'LowerThanMaxLimit': 30,
    'DecreasingLimit': 12,

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

    'WeightIniter': 'XavierNorm',

    # Params for NormWeightIniter only
    'InitMean' : 0,
    'InitStd' : 1,

    'AtomFeatureSize': 39,
    'BondFeatureSize': 10,

    'GCNReadout': 'Add',


    'SplitValidSeed': 8,
    'SplitTestSeed': 8,
    'BatchSize': 200,

}
AdjustableParamList = {}
SpecificParamList = {
    'DropRate':[0.2],
    'WeightDecay':[5],
    'lr':[4],
    'GCNInputSize': [128],
    'GCNHiddenSize': [512],
    'GCNLayers': [2],
    'FPSize':[256],
    'DNNLayers':[[128, 32]],
}

expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()
