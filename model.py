
#TODO STEP 1: Define the GNN with attention mechanism

#TODO STEP 2: Define the RL enviroment   

#TODO STEP 3: Train the GNN

#TODO STEP 4: Train the RL Agent

#TODO STEP 5: Evaluate

import torch
import torch.nn as nn

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNWithAttention, self).__init__()

        self.gat1 = nn.Sequential(
            GATConv(input_dim, hidden_dim, heads=1, concat=True),
            nn.ReLU()
        )

        self.gat2 = nn.Sequential(
            GATConv(hidden_dim, output_dim, heads=1, concat=True),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x, (edge_index, attention_weights) = self.gat2(x, edge_index, return_attention_weights=True)

        return x, attention_weights