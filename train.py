import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from model import GNNWithAttention

# Load the CORA dataset
dataset = Planetoid(root="data/CORA", name="CORA")
graph_data = dataset[0]

print("Graph data:")
print(f"Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}")
print(f"Features shape: {graph_data.x.shape}, Labels: {graph_data.y.unique()}")

in_channels = graph_data.num_node_features
hidden_channels = 16
out_channels = dataset.num_classes  # For CORA, it's 7

model = GNNWithAttention(in_channels, hidden_channels, out_channels)
print(model)
