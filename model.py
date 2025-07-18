import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool

class GNNWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_channels, dropout = 0.2, heads = 1):
        super(GNNWithAttention, self).__init__()

        # GAT Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * heads)
        
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * heads)

        #Dropout
        self.dropout = nn.Dropout(dropout)

        self.skip_connection1 = nn.Linear(input_dim, hidden_dim * heads)

        # Fully connected classification layer
        self.fc = nn.Linear(hidden_dim * heads, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels) 

    def forward(self, data, return_attention=False, return_embeddings=False):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Layer 1: GAT with residual connection ---
        x_res1 = x
        x1 = self.gat1(x, edge_index)
        x1 = self.layer_norm1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        x1 = x1 + self.skip_connection1(x_res1)
        
        # --- Layer 2: GAT with residual connection ---
        x_res2 = x1
        
        if return_attention:
            x2, (edge_index_att, att_weights) = self.gat2(
                x1, edge_index, return_attention_weights=True
            )
        else:
            x2 = self.gat2(x1, edge_index)
           
        x2 = self.layer_norm2(x2 + x_res2)  # Residual connection
        node_embeddings = F.elu(x2)
        node_embeddings = self.dropout(node_embeddings)
        
        # --- Global pooling and classification ---
        graph_emb = global_mean_pool(node_embeddings, batch)
        logits = self.fc(graph_emb)
        logits = self.batch_norm(logits)
        
        # --- Return handling ---
        if return_embeddings:
            if return_attention:
                return logits, node_embeddings, (edge_index_att, att_weights)
            return logits, node_embeddings
        else:
            if return_attention:
                return logits, (edge_index_att, att_weights)
            return logits