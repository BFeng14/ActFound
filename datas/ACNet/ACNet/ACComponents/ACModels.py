import torch as t
import torch.nn as nn
from sklearn import svm
from Models.CMPNN.CMPNNModel import *
from Models.BasicGNNs import *
from Models.Graphormer.Graphormer import Graphormer
from Models.ClassifierModel import DNN


class ACPredMLP(nn.Module):
    def __init__(self, opt):
        super(ACPredMLP, self).__init__()
        self.opt = opt
        # todo(zqzhang): updated in ACv8
        if self.opt.args['Feature'] == 'FP':
            self.input_size = self.opt.args['nBits']
        elif self.opt.args['Feature'] == 'Raw':
            self.input_size = self.opt.args['RawFeatureSize']
        self.Classifier = DNN(
                input_size = 2 * self.input_size,
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        Input1, Input2 = Input
        Input1 = Input1.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        Input2 = Input2.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        # print("Input1:")
        # print(Input1)
        # print(Input1.size())
        # print('Input2:')
        # print(Input2)
        # print(Input2.size())
        # [batch_size, nBits]
        PairwiseMolFeature = t.cat([Input1, Input2], dim=1)
        # print("PairwiseFeature:")
        # print(PairwiseMolFeature)
        # print(PairwiseMolFeature.size())
        prediction = self.Classifier(PairwiseMolFeature)
        # print("Prediction:")
        # print(prediction)
        # print(prediction.size())

        return prediction

class ACPredLSTM(nn.Module):
    def __init__(self, opt):
        super(ACPredLSTM, self).__init__()
        self.opt = opt
        self.WordEmbed = nn.Embedding(self.opt.args['MaxDictLength'],
                                      self.opt.args['FPSize'],
                                      padding_idx = self.opt.args['MaxDictLength']-1)
        self.MolFeatureExtractor = nn.LSTM(input_size = self.opt.args['FPSize'],
                                           hidden_size = self.opt.args['FPSize'],
                                           num_layers = self.opt.args['LSTMLayers'],
                                           batch_first = True,
                                           bidirectional = True

        )
        self.Classifier = DNN(
                input_size = 2*self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        Input1, Input2 = Input
        Input1 = Input1.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        Input2 = Input2.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        Embed1 = self.WordEmbed(Input1)
        Embed2 = self.WordEmbed(Input2)
        _, (MolFeature1,_) = self.MolFeatureExtractor(Embed1)
        _, (MolFeature2,_) = self.MolFeatureExtractor(Embed2)
        # print(MolFeature1)
        # print(MolFeature1.size())
        # MolFeature: [ LSTMLayer*Bi, Batch_size, FP_size]
        MolFeature1 = MolFeature1.permute(1,0,2)
        MolFeature2 = MolFeature2.permute(1,0,2)
        # MolFeature: [Batch_size, LSMTLayer*Bi, FP_size]
        MolFeature1 = MolFeature1.sum(dim=1)
        MolFeature2 = MolFeature2.sum(dim=1)
        # MolFeature: [Batch_size, FP_size]
        PairwiseMolFeature = t.cat([MolFeature1,MolFeature2],dim=1)
        prediction = self.Classifier(PairwiseMolFeature)
        return prediction

class ACPredGRU(nn.Module):
    def __init__(self, opt):
        super(ACPredGRU, self).__init__()
        self.opt = opt
        self.WordEmbed = nn.Embedding(self.opt.args['MaxDictLength'],
                                      self.opt.args['FPSize'],
                                      padding_idx = self.opt.args['MaxDictLength'] - 1)
        self.MolFeatureExtractor = nn.GRU(input_size = self.opt.args['FPSize'],
                                          hidden_size = self.opt.args['FPSize'],
                                          num_layers = self.opt.args['GRULayers'],
                                          batch_first = True,
                                          bidirectional = True)
        self.Classifier = DNN(
                input_size = 2 * self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        Input1, Input2 = Input
        Input1 = Input1.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        Input2 = Input2.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        Embed1 = self.WordEmbed(Input1)
        Embed2 = self.WordEmbed(Input2)
        _, MolFeature1 = self.MolFeatureExtractor(Embed1)
        _, MolFeature2 = self.MolFeatureExtractor(Embed2)
        # print(MolFeature1)
        # print(MolFeature1.size())
        # MolFeature: [ GRULayer*Bi, Batch_size, FP_size]
        MolFeature1 = MolFeature1.permute(1, 0, 2)
        MolFeature2 = MolFeature2.permute(1, 0, 2)
        # MolFeature: [Batch_size, GRULayer*Bi, FP_size]
        MolFeature1 = MolFeature1.sum(dim = 1)
        MolFeature2 = MolFeature2.sum(dim = 1)
        # MolFeature: [Batch_size, FP_size]
        PairwiseMolFeature = t.cat([MolFeature1, MolFeature2], dim = 1)
        prediction = self.Classifier(PairwiseMolFeature)
        return prediction

class ACPredCMPNN(nn.Module):
    def __init__(self, opt):
        super(ACPredCMPNN, self).__init__()
        self.opt = opt
        self.MolFeatureExtractor = CMPNNModel(
                    self.opt.args['dataset_type']=='classification',
                    self.opt.args['dataset_type']=='multiclass',
                    opt = self.opt)
        self.Classifier = DNN(
                input_size = 2 * self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        Input1, Input2 = Input
        MolFeature1 = self.MolFeatureExtractor(Input1)
        MolFeature2 = self.MolFeatureExtractor(Input2)
        # print(f"size of Mol1 and Mol2: {MolFeature1.size()}")
        PairwiseMolFeature = t.cat([MolFeature1,MolFeature2],dim=1)
        # print(f'size of PairwiseMolFeature: {PairwiseMolFeature.size()}')
        prediction = self.Classifier(PairwiseMolFeature)

        return prediction

class ACPredGCN(nn.Module):
    def __init__(self, opt):
        super(ACPredGCN, self).__init__()
        self.opt = opt
        if not self.opt.args['PyG']:
            print(f"PyG arg should be {True}")
            raise ValueError
        self.MolFeatureExtractor = PyGGCN(self.opt, FeatureExtractor = True)
        self.Classifier = DNN(
                input_size = 2 *self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        self.reset_batch(Input)
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        # print(f"Input.batch: {Input.batch}")
        MolEmbeddings = self.MolFeatureExtractor(Input)
        MolEmbeddings = self.decompose_mol_pair(MolEmbeddings)

        prediction = self.Classifier(MolEmbeddings)
        return prediction

    def reset_batch(self, Input):
        batch = Input.batch
        atom_nums = Input.atom_num
        bond_nums = Input.bond_num
        MolNum = len(atom_nums)
        # print(f"batch: {batch}")
        # print(f"atom_nums: {atom_nums}")
        # print(f"MolNum: {MolNum}")
        # print(f"len batch: {len(batch)}")
        # print(f"sum atom_nums: {t.sum(atom_nums)}")
        assert len(batch) == t.sum(atom_nums)

        # reset batch by atom num
        mol_cnt = 0
        mol_batch = t.Tensor([])
        for i in range(MolNum):
            tmp = t.Tensor([mol_cnt])
            tmp = tmp.repeat(atom_nums[i].item())
            assert len(tmp) == atom_nums[i]
            mol_batch = t.cat([mol_batch, tmp]).long()
            mol_cnt += 1
        Input.batch = mol_batch

    def decompose_mol_pair(self, MolEmbeddings):
        # print(f"MolEmbedding size: {MolEmbeddings.size()}")
        mol_num = MolEmbeddings.size()[0]
        EmbLength = MolEmbeddings.size()[1]
        assert mol_num % 2 == 0
        return MolEmbeddings.view(int(mol_num/2), int(EmbLength*2))

class ACPredGIN(nn.Module):
    def __init__(self, opt):
        super(ACPredGIN, self).__init__()
        self.opt = opt
        if not self.opt.args['PyG']:
            print(f"PyG arg should be {True}")
            raise ValueError
        self.MolFeatureExtractor = PyGGIN(self.opt, FeatureExtractor = True)
        self.Classifier = DNN(
                input_size = 2 *self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        self.reset_batch(Input)
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        # print(f"Input.batch: {Input.batch}")
        MolEmbeddings = self.MolFeatureExtractor(Input)
        MolEmbeddings = self.decompose_mol_pair(MolEmbeddings)

        prediction = self.Classifier(MolEmbeddings)
        return prediction

    def reset_batch(self, Input):
        batch = Input.batch
        atom_nums = Input.atom_num
        bond_nums = Input.bond_num
        MolNum = len(atom_nums)
        # print(f"batch: {batch}")
        # print(f"atom_nums: {atom_nums}")
        # print(f"MolNum: {MolNum}")
        # print(f"len batch: {len(batch)}")
        # print(f"sum atom_nums: {t.sum(atom_nums)}")
        assert len(batch) == t.sum(atom_nums)

        # reset batch by atom num
        mol_cnt = 0
        mol_batch = t.Tensor([])
        for i in range(MolNum):
            tmp = t.Tensor([mol_cnt])
            tmp = tmp.repeat(atom_nums[i].item())
            assert len(tmp) == atom_nums[i]
            mol_batch = t.cat([mol_batch, tmp]).long()
            mol_cnt += 1
        Input.batch = mol_batch

    def decompose_mol_pair(self, MolEmbeddings):
        # print(f"MolEmbedding size: {MolEmbeddings.size()}")
        mol_num = MolEmbeddings.size()[0]
        EmbLength = MolEmbeddings.size()[1]
        assert mol_num % 2 == 0
        return MolEmbeddings.view(int(mol_num/2), int(EmbLength*2))

class ACPredSGC(nn.Module):
    def __init__(self, opt):
        super(ACPredSGC, self).__init__()
        self.opt = opt
        if not self.opt.args['PyG']:
            print(f"PyG arg should be {True}")
            raise ValueError
        self.MolFeatureExtractor = PyGSGC(self.opt, FeatureExtractor = True)
        self.Classifier = DNN(
                input_size = 2 *self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        self.reset_batch(Input)
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        # print(f"Input.batch: {Input.batch}")
        MolEmbeddings = self.MolFeatureExtractor(Input)
        MolEmbeddings = self.decompose_mol_pair(MolEmbeddings)

        prediction = self.Classifier(MolEmbeddings)
        return prediction

    def reset_batch(self, Input):
        batch = Input.batch
        atom_nums = Input.atom_num
        bond_nums = Input.bond_num
        MolNum = len(atom_nums)
        # print(f"batch: {batch}")
        # print(f"atom_nums: {atom_nums}")
        # print(f"MolNum: {MolNum}")
        # print(f"len batch: {len(batch)}")
        # print(f"sum atom_nums: {t.sum(atom_nums)}")
        assert len(batch) == t.sum(atom_nums)

        # reset batch by atom num
        mol_cnt = 0
        mol_batch = t.Tensor([])
        for i in range(MolNum):
            tmp = t.Tensor([mol_cnt])
            tmp = tmp.repeat(atom_nums[i].item())
            assert len(tmp) == atom_nums[i]
            mol_batch = t.cat([mol_batch, tmp]).long()
            mol_cnt += 1
        Input.batch = mol_batch

    def decompose_mol_pair(self, MolEmbeddings):
        # print(f"MolEmbedding size: {MolEmbeddings.size()}")
        mol_num = MolEmbeddings.size()[0]
        EmbLength = MolEmbeddings.size()[1]
        assert mol_num % 2 == 0
        return MolEmbeddings.view(int(mol_num/2), int(EmbLength*2))

class ACPredGraphormer(nn.Module):
    def __init__(self, opt):
        super(ACPredGraphormer, self).__init__()
        self.opt = opt
        self.MolFeatureExtractor = Graphormer(
                num_encoder_layers = self.opt.args['num_encoder_layers'],
                num_attention_heads = self.opt.args['num_attention_heads'],
                embedding_dim = self.opt.args['embedding_dim'],
                dropout_rate = self.opt.args['dropout_rate'],
                intput_dropout_rate = self.opt.args['intput_dropout_rate'],
                ffn_dim = self.opt.args['ffn_dim'],
                edge_type = self.opt.args['edge_type'],
                multi_hop_max_dist = self.opt.args['multi_hop_max_dist'],
                attention_dropout_rate = self.opt.args['attention_dropout_rate'],
                flag = self.opt.args['flag'],
                opt = self.opt,
                mode = 'Extractor'
        )
        self.Classifier = DNN(
                input_size = 2 * self.opt.args['embedding_dim'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt
        )

    def forward(self, Input):
        Input1, Input2 = Input
        MolFeature1 = self.MolFeatureExtractor(Input1)
        MolFeature2 = self.MolFeatureExtractor(Input2)
        PairwiseMolFeature = t.cat([MolFeature1, MolFeature2], dim=1)
        prediction = self.Classifier(PairwiseMolFeature)

        return prediction

