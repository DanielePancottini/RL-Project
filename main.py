from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

from train_gat import TrainGAT

# Load the CORA dataset
dataset = Planetoid(root="data/CORA", name="CORA")
graph_data = dataset[0]

print("Graph data:")
print(f"Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}")
print(f"Features shape: {graph_data.x.shape}, Labels: {graph_data.y.unique()}")

# Train GNN

trainer = TrainGAT(dataset, graph_data)
trained_model = trainer.train_gat()

# Evaluate the model on the test set
test_accuracy = trainer.evaluate_accuracy(trained_model, dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')