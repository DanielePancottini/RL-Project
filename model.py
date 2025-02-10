import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_channels, dropout = 0.2, heads = 1):
        super(GNNWithAttention, self).__init__()

        # GAT Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        #Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected classification layer
        self.fc = nn.Linear(hidden_dim, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels) 

    def forward(self, data, return_attention=False):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT Layer
        x = F.relu(self.gat1(x, edge_index))
        x = self.layer_norm1(x)
        x = self.dropout(x)

        # Second GAT Layer
        if return_attention:
            x, attention_weights = self.gat2(x, edge_index, return_attention_weights=True)
            x = self.layer_norm2(x)
        else:
            x = self.gat2(x, edge_index)
            x = self.layer_norm2(x)

        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        logits = self.fc(x)
        logits = self.batch_norm(logits)

        if return_attention:
            return logits, attention_weights
        return logits