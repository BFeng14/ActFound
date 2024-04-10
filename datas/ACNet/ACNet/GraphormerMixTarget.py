from ACComponents.ACProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SubsetsNum': 1,
    'OnlyEval': False,
}

BasicParamList = {
    'ExpName': 'ACMixTarget',
    'MainMetric': 'AUC',
    'DataPath': './ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Mixed_Screened.json',
    'RootPath': './TestExp/Mix/Graphormer/',
    'CUDA_VISIBLE_DEVICES': '3',
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,
    'Feature': 'Graphormer',
    'Model': 'Graphormer',

    'OnlySpecific': True,
    'Weight': True,
    'AC': True,
    'PyG': False,

    'ValidRate': 40000,
    'PrintRate': 5,
    'UpdateRate': 1,
    'SplitRate': [0.8, 0.1],
    'Splitter': 'TargetRandom',
    'MaxEpoch': 300,
    'LowerThanMaxLimit': 12,
    'DecreasingLimit': 8,

    # if OnlyEval == True:
    'EvalModelPath': None,
    'EvalDatasetPath': None,
    'EvalLogAllPreds': None,

    'Scheduler': 'PolynomialDecayLR',
    # 'Scheduler': 'EmptyLRScheduler',


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

    'FeatureCategory': 'BaseED',

    # Params for Graphormer only
    'num_offset': 16,
    'num_atoms': 16 * 8,      # offset * AtomFeatureNum
    'num_in_degree': 16,       # length of indegree dictionary
    'num_out_degree': 16,      # length of outdegree dictionary
    'num_edges': 16 * 4,      # offset * BondFeatureNum
    'num_spatial': 512,         # length of SPD dictionary, must be larger than the largest SPD
    'num_edge_dis': 30,         # must be larger than multi-hop-max-dist
    'dropout_rate': 0.1,
    'intput_dropout_rate': 0.1,
    'edge_type': 'multi_hop',
    'multi_hop_max_dist': 20,
    'flag': False,
    'spatial_pos_max': 20,
    'max_node': 512,

    # Training Params to be adujsted. If the param is not needed to be adjusted, set the value here.
    'SplitValidSeed': 8,
    'SplitTestSeed': 8,
    'BatchSize': 32,

}
AdjustableParamList = {}
SpecificParamList = {
    'DropRate':[0.4],
    'WeightDecay':[5],
    'lr':[4],
    'num_encoder_layers':[10],
    'num_attention_heads':[32],
    'embedding_dim':[256],
    'ffn_dim':[256],
    'attention_dropout_rate':[0.4],
    'DNNLayers':[[64, 16]],
}


expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

