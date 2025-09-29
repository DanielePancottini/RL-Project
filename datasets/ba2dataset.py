import pickle as pkl
import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

class BA2Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # leave empty since we handle loading manually

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
        with open(path, 'rb') as fin:
            adjs, features, labels = pkl.load(fin)

        data_list = []

        for i in range(len(adjs)):
            adj = adjs[i]
            x = torch.tensor(features[i], dtype=torch.float)
            y = torch.tensor(np.argmax(labels[i]), dtype=torch.long)

            row, col = torch.tensor(adj.nonzero(), dtype=torch.long)
            edge_index = torch.stack([row, col], dim=0)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
