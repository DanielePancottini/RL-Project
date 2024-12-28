import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_classes=2):
        super(GNNWithAttention, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        self.fc = nn.Linear(output_dim, num_classes)  # Fully connected layer for classification

    def forward(self, x, edge_index, return_attention=False):
        # Apply GAT layers
        x = F.elu(self.gat1(x, edge_index))
        if return_attention:
            x, (edge_index, attention_weights) = self.gat2(x, edge_index, return_attention_weights=True)
            return x, (edge_index, attention_weights)
        else:
            x = self.gat2(x, edge_index)
        
        # Classification head
        logits = self.fc(x)
        return logits