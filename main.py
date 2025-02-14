import torch

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from model import GNNWithAttention

from sklearn.model_selection import train_test_split

from train_gat import Trainer

from collections import Counter

import numpy as np

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load HIV dataset
dataset = MoleculeNet(root='./data', name='HIV')

# Check dataset details
print(f'Dataset size: {len(dataset)} | Dataset classes: {dataset.num_classes} | Target shape: {dataset.y.shape}')
print(dataset[0])  # Print details of the first molecule graph

"""
    Upsampling Section
"""

# Assuming `data.y` contains the class labels
labels = dataset.data.y.squeeze().cpu().numpy()

# Assuming `dataset` is your original dataset and `labels` are the class labels
minority_class = 1.0  # Active molecules
majority_class = 0.0  # Inactive molecules

# Get indices of minority and majority classes
minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
majority_indices = [i for i, label in enumerate(labels) if label == majority_class]

# Create subsets for minority and majority classes
minority_dataset = dataset[torch.tensor(minority_indices, dtype=torch.long)]
majority_dataset = dataset[torch.tensor(majority_indices, dtype=torch.long)]

# Calculate how many times to duplicate the minority class
num_minority = len(minority_dataset)  # Number of samples in minority class
num_majority = len(majority_dataset)  # Number of samples in majority class
upsample_factor = num_majority // num_minority  # How many times to duplicate

# Duplicate the minority class samples
upsampled_minority_dataset = torch.utils.data.ConcatDataset([minority_dataset] * upsample_factor)

# Combine the upsampled minority class with the majority class
balanced_dataset = upsampled_minority_dataset + majority_dataset

# Update labels after upsampling
balanced_labels = [data.y.item() for data in balanced_dataset]

# Count class occurrences
class_counts = Counter(balanced_labels)

# Class weights 
class_weights = torch.tensor([len(dataset)/(2*count) for _, count in class_counts.items()]).to(device)

# Print class balances in one line
print("HIV Class Balances: " + ", ".join([f"Class {cls}: {count}" for cls, count in class_counts.items()]))
print(f"Class Weights: {class_weights}")

"""
    Random Splitting Section
"""

# Train-Test split (80% train, 10% validation, 10% test)
indices = np.arange(len(balanced_dataset))

# Split indices into training (80%), validation (10%), and test (10%) sets
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

# Step 3: Create DataLoader for each dataset
train_dataset = [balanced_dataset[i] for i in train_indices]
val_dataset = [balanced_dataset[i] for i in valid_indices]
test_dataset = [balanced_dataset[i] for i in test_indices]

"""
    Input Normalization Section
"""

# Compute mean and std **only from training data**
all_x_train = torch.cat([data.x.float() for data in train_dataset], dim=0)
mean = all_x_train.mean(dim=0, keepdim=True)
std = all_x_train.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero

# Normalize train and test datasets using **training** statistics
def normalize_dataset(dataset, mean, std):
    for data in dataset:
        data.x = (data.x.float() - mean) / std
    
    return dataset

train_dataset = normalize_dataset(train_dataset, mean, std)
val_dataset = normalize_dataset(val_dataset, mean, std)
test_dataset = normalize_dataset(test_dataset, mean, std)

# Check if normalization was applied
print("Normalized train dataset first example:", train_dataset[0], train_dataset[0].x)

"""
    Data Loaders Section
"""

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
    Model Section
"""

# Model Parameters and Model Definition
in_channels = dataset.num_node_features
hidden_channels = 256
num_classes = dataset.num_classes
heads = 8
dropout = 0.3

# Model & Optimizer
model = GNNWithAttention(in_channels, hidden_channels, num_classes, dropout, heads).to(device)

"""
    Training Section
"""

# Train GNN
trainer = Trainer(model, class_weights, device)
trained_model = trainer.train(train_loader, val_loader)

# Save the model
torch.save(trained_model.state_dict(), "./models/trained_model.pt")

"""
    Evaluation Section
"""

# Evaluate the model on the test set
accuracy, precision, recall, f1, conf_matrix, auc_roc = trainer.evaluate_metrics(test_loader)

# Print metrics
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print(f"AUC-ROC: {auc_roc}")