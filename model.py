import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_channels, heads=1):
        super(GNNWithAttention, self).__init__()

        # GAT Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

        # Fully connected classification layer
        self.fc = nn.Linear(hidden_dim, out_channels)

    def forward(self, data, return_attention=False):

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # Apply GAT layers with ReLU activation
        x = F.relu(self.gat1(x, edge_index))
        
        if return_attention:
            x, attention_weights = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = F.relu(self.gat2(x, edge_index))

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Classification head
        logits = self.fc(x)

        if return_attention:
            return logits, attention_weights
        return logits