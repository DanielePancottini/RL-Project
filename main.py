import torch
from datasets.ba2dataset import BA2Dataset
from datasets.ground_truth_loader import generate_expert_data_from_ground_truth
from load_gcn import load_gcn_checkpoint
from torch_geometric.loader import DataLoader
from env import GNNInterpretEnvironment
from policy import Policy, pretrain_policy
from model import GCNModel
from sklearn.model_selection import train_test_split
from train_model import Trainer
from collections import Counter
import numpy as np
from policy import train_reinforce_rollout
import os

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load HIV dataset
#dataset = MoleculeNet(root='./data', name='HIV')
dataset = BA2Dataset(root='./data/ba2')

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

# Class weights for loss function, calculated based on the balanced dataset
class_weights_list = [0.0] * dataset.num_classes # Initialize for all possible classes
for cls, count in class_counts.items():
    if count > 0:
        # Use len(balanced_dataset) for the total count in the balanced dataset
        class_weights_list[int(cls)] = len(balanced_dataset) / (dataset.num_classes * count)

class_weights = torch.tensor(class_weights_list).to(device)

# Print class balances in one line
print("Dataset Class Balances: " + ", ".join([f"Class {cls}: {count}" for cls, count in class_counts.items()]))
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

"""
train_dataset = normalize_dataset(train_dataset, mean, std)
val_dataset = normalize_dataset(val_dataset, mean, std)
test_dataset = normalize_dataset(test_dataset, mean, std)
"""

"""
    Data Loaders Section
"""

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
    TEST
"""
first_batch = next(iter(train_loader))
print(first_batch)

"""
    Model Section
"""

# Model Parameters and Model Definition
features_dim = dataset.num_node_features
num_classes = dataset.num_classes

"""
    Training Section
"""

model_path = "./models/ba2/best_model"

# Model Definition
model = GCNModel(features_dim, num_classes, device).to(device)

# If already trained skip training
if not os.path.exists(model_path):

    # Train GNN
    trainer = Trainer(model, class_weights, device)
    trained_model = trainer.train(train_loader, val_loader)

    # Save the model
    #torch.save(trained_model.state_dict(), "./models/trained_model.pt")

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

model, info = load_gcn_checkpoint(model, model_path, device)

"""
    RL Agent Training
"""

# Define PPO DataLoaders
env_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)

model.eval()

# Assuming you have your baseline GNN model and data
baseline_gnn = model

env = GNNInterpretEnvironment(gnn_model=baseline_gnn, dataloader=env_dataloader, max_steps=20, device=device)

# policy
input_dim = features_dim + 2  # original features + start flag + in-S flag
policy = Policy(input_dim=input_dim, hidden_dim=64, L=3, alpha=0.85, device=device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

# 1. Generate the expert data using the ground truth loader
# NOTE: Make sure your 'train_dataset' is shuffled in the same way as the
# ground truth loader (i.e., using a fixed seed like 42).
# Your train_test_split uses random_state=42, which is perfect.
expert_trajectories = generate_expert_data_from_ground_truth(balanced_dataset, train_indices, device)

# 2. Run the pre-training using the same function as before
#pretrain_policy(policy, optimizer, expert_trajectories, env, epochs=10, device=device)

print("\n" + "="*50)
print("GROUND TRUTH PRE-TRAINING COMPLETE. STARTING REINFORCEMENT LEARNING.")
print("="*50 + "\n")

# training hyperparameters
EPISODES = 300
ROLLOUT_M = 5            # Monte-Carlo rollouts per intermediate step (paper uses M)
ROLLOUT_MAX_STEPS = 20
USE_BASELINE = False
ENTROPY = 1e-6

train_reinforce_rollout(env, policy, optimizer,
                        episodes=EPISODES,
                        rollout_M=ROLLOUT_M,
                        rollout_max_steps=ROLLOUT_MAX_STEPS,
                        baseline=USE_BASELINE,
                        entropy_coeff=ENTROPY,
                        device=device,
                        log_every=10)
