from ACComponents.ACProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SubsetsNum': 13,
    'OnlyEval': False,
    'Finetune':True,
}

BasicParamList = {
    'ExpName': 'ACFew',
    'MainMetric': 'AUC',
    'DataPath': './ACComponents/ACDataset/data_files/MMP_AC_Few_representation/GraphLoG.npz',
    'RootPath': './TestExp/Few/GraphLoG/',
    'CUDA_VISIBLE_DEVICES': '2',
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,
    'Feature': 'Raw',
    'Model': 'MLP',

    # if Feature == Raw
    'RawFeatureSize': 300,

    'OnlySpecific': True,
    'Weight': True,
    'AC': True,
    'PyG': False,

    'ValidRate': 40000,
    'PrintRate': 5,
    'UpdateRate': 1,
    'ValidBalance': False,
    'TestBalance': False,
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

    # Training Params to be adujsted. If the param is not needed to be adjusted, set the value here.
    'SplitValidSeed': 8,
    'SplitTestSeed': 8,
    'BatchSize': 8,

}
AdjustableParamList = {}
SpecificParamList = {
    'DropRate':[0.2],
    'WeightDecay':[5],
    'lr':[4],
    'DNNLayers':[[256,64]],
}


expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

