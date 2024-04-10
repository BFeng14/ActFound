from TrainingFramework.FileUtils import *
from TrainingFramework.Splitter import *
from TrainingFramework.Featurizer import *
from torch.utils import data
from torch_geometric.data import InMemoryDataset
import os

class PyGMolDataset(InMemoryDataset):
    def __init__(self, graphdataset, opt, mode):
        self.graph_dataset = graphdataset
        self.opt = opt
        # todo(zqzhang): updated in ACv7
        self.dataset_path_root = self.opt.args['ExpDir'] + 'Dataset/'
        if not os.path.exists(self.dataset_path_root):
            os.mkdir(self.dataset_path_root)
        self.mode = mode
        if os.path.exists(self.dataset_path_root + 'processed/' + self.processed_file_names[0]):
            os.remove(self.dataset_path_root + 'processed/' + self.processed_file_names[0])
        super(PyGMolDataset, self).__init__(root = self.dataset_path_root)
        self.data, self.slices = t.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.opt.args['DataPath']]

    @property
    def processed_file_names(self):
        return [self.opt.args['ExpName'] + '_' + self.mode + '.pt']

    def download(self):
        pass

    def process(self):
        data_list = self.graph_dataset
        data, slices = self.collate(data_list)
        # print("Processed without saving complete.")
        print("Saving processed files...")
        t.save((data, slices), self.processed_paths[0])
        print('Saving complete!')

    # def __len__(self):
    #     return len(self.graph_dataset)
