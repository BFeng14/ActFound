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
    'DataPath': './ACComponents/ACDataset/data_files/MMP_AC_Few_representation/PretrainGNNs.npz',
    'RootPath': './TestExp/Few/PretrainGNNs/',
    'CUDA_VISIBLE_DEVICES': '3',
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
    'WeightDecay':[4.5],
    'lr':[3],
    'DNNLayers':[[128]],
}


expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

