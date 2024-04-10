from ACComponents.ACProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SubsetsNum': 64,
    'OnlyEval': False,
}

BasicParamList = {
    'ExpName': 'ACMedium',
    'MainMetric': 'AUC',
    'DataPath': './ACComponents/ACDataset/data_files/generated_datasets/MMP_AC_Medium.json',
    'RootPath': './TestExp/Medium/Graphormer/',
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

    'FeatureCategory': 'BaseOH',

    # Params for Graphormer only
    'num_offset': 16,
    'num_atoms': 16 * 39,      # offset * AtomFeatureNum
    'num_in_degree': 16,       # length of indegree dictionary
    'num_out_degree': 16,      # length of outdegree dictionary
    'num_edges': 16 * 10,      # offset * BondFeatureNum
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
    'DropRate':[0.2],
    'WeightDecay':[4.5],
    'lr':[3],
    'num_encoder_layers':[4],
    'num_attention_heads':[8],
    'embedding_dim':[32],
    'ffn_dim':[32],
    'attention_dropout_rate':[0.1],
    'DNNLayers':[[]],
}


expcontroller = ACExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

