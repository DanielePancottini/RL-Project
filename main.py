import torch

from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
from model import GNNWithAttention

from train_gat import Trainer

from rdkit import Chem
from rdkit.Chem import Draw

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CORA dataset
dataset = MoleculeNet(root="data/Tox21", name="Tox21")
graph_data = dataset[0]

# Check dataset details
sample = dataset[0] # View the first molecule graph
print(sample)

# Convert SMILES to molecule structure
smiles = sample['smiles']
mol = Chem.MolFromSmiles(smiles)

# Draw the molecule
Draw.MolToImage(mol).show()

"""
# Define DataLoaders

# Train-Test split (80% train, 20% test)
num_train = int(0.8 * len(dataset))
num_test = len(dataset) - num_train
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

# Data loaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model Parameters and Model Definition
in_channels = dataset.num_node_features
hidden_channels = 128
num_classes = dataset.num_classes

# Model & Optimizer
model = GNNWithAttention(in_channels, hidden_channels, num_classes).to(device)

# Train GNN

trainer = Trainer(model, device)
trained_model = trainer.train(train_loader, test_loader)

# Evaluate the model on the test set
test_accuracy = trainer.evaluate_accuracy(trained_model, dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')
"""