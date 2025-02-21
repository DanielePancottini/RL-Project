import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class GNNInterpretEnvironment(gym.Env):
    def __init__(self, gnn_model, dataloader, batch_size=32, device='cuda'):
        super().__init__()
        self.gnn_model = gnn_model
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.device = device
        
        # Get max nodes/edges across over the possible batches
        # We can define fixed action and obeservation spaces

        self.max_nodes, self.max_edges = get_max_nodes_edges(dataloader)
       
        # Action space: continuous values between 0 and 1 for each node and edge
        # Will be truncated to match actual batch size during step()
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_nodes + self.max_edges, ),
            dtype=np.float32
        )
        
        # Take a batch to extract the number of features
        example_batch = next(self.data_iter)

        # Observation space matches PyG's batch format
        self.observation_space = spaces.Dict({
            'x': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_nodes, example_batch.x.shape[1]),
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0,
                high=self.max_nodes,
                shape=(2, self.max_edges),
                dtype=np.int64
            ),
            'batch': spaces.Box(
                low=0,
                high=batch_size,
                shape=(self.max_nodes,),
                dtype=np.int64
            )
        })
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Get next batch
        try:
            self.current_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            self.current_batch = next(self.data_iter)
        
        self.current_batch = self.current_batch.to(self.device)

        # Pad data to max_nodes / max_edges
        x_padded = torch.zeros((self.max_nodes, self.current_batch.x.size(1)), device=self.device)
        x_padded[:self.current_batch.x.size(0)] = self.current_batch.x

        edge_index_padded = torch.zeros((2, self.max_edges), dtype=torch.int64, device=self.device)
        edge_index_padded[:, :self.current_batch.edge_index.size(1)] = self.current_batch.edge_index

        batch_padded = torch.zeros((self.max_nodes,), dtype=torch.int64, device=self.device)
        batch_padded[:self.current_batch.batch.size(0)] = self.current_batch.batch

        observation = {
            'x': x_padded.cpu().numpy(),
            'edge_index': edge_index_padded.cpu().numpy(),
            'batch': batch_padded.cpu().numpy()
        }

        info = {
            'num_graphs': self.current_batch.num_graphs,
            'num_nodes': self.current_batch.x.size(0),
            'num_edges': self.current_batch.edge_index.size(1)
        }

        return observation, info
    
    def step(self, action):
        # Get current batch sizes
        num_nodes = self.current_batch.x.size(0)
        num_edges = self.current_batch.edge_index.size(1)
        
        # Truncate action to match current batch
        node_mask = torch.tensor(action[:num_nodes], device=self.device)
        edge_mask = torch.tensor(action[num_nodes:num_nodes + num_edges], device=self.device)
        
        # Apply masks and compute predictions for the batch
        masked_x = self.current_batch.x * node_mask.unsqueeze(-1)
        masked_edge_index = self.current_batch.edge_index[:, edge_mask > 0.5]
        
        # Create masked batch
        masked_batch = Data(
            x=masked_x,
            edge_index=masked_edge_index,
            batch=self.current_batch.batch
        )
        
        # Get predictions
        with torch.no_grad():
            original_pred = self.gnn_model(self.current_batch)
            masked_pred = self.gnn_model(masked_batch)
        
        # Compute reward
        pred_similarity = -F.mse_loss(masked_pred, original_pred)
        sparsity = -(node_mask.mean() + edge_mask.mean()) * 0.1
        reward = pred_similarity + sparsity
        
        # Get next batch
        try:
            self.current_batch = next(self.data_iter)
        except StopIteration:
            print("EXCEPTION")
            self.data_iter = iter(self.dataloader)
            self.current_batch = next(self.data_iter)
            
        self.current_batch = self.current_batch.to(self.device)
        print(f"Graph has {self.current_batch.x.size(0)} nodes")

        
        # Pad next batch
        x_padded = torch.zeros((self.max_nodes, self.current_batch.x.size(1)), device=self.device)
        x_padded[:self.current_batch.x.size(0)] = self.current_batch.x

        edge_index_padded = torch.zeros((2, self.max_edges), dtype=torch.int64, device=self.device)
        edge_index_padded[:, :self.current_batch.edge_index.size(1)] = self.current_batch.edge_index

        batch_padded = torch.zeros((self.max_nodes,), dtype=torch.int64, device=self.device)
        batch_padded[:self.current_batch.batch.size(0)] = self.current_batch.batch

        observation = {
            'x': x_padded.cpu().numpy(),
            'edge_index': edge_index_padded.cpu().numpy(),
            'batch': batch_padded.cpu().numpy()
        }

        info = {
            'num_graphs': self.current_batch.num_graphs,
            'num_nodes': self.current_batch.x.size(0),
            'num_edges': self.current_batch.edge_index.size(1),
            'pred_similarity': pred_similarity.item(),
            'sparsity': sparsity.item()
        }

        return observation, reward.item(), False, False, info

""" Computes the max number of nodes and edges in any batch of the dataset. """
def get_max_nodes_edges(dataloader):
    max_nodes = 0
    max_edges = 0

    for batch in dataloader:
        print(f"Batch has {batch.x.size(0)} nodes, {batch.edge_index.size(1)} edges")
        num_nodes = batch.x.size(0)
        num_edges = batch.edge_index.size(1)

        max_nodes = max(max_nodes, num_nodes)
        max_edges = max(max_edges, num_edges)

    return max_nodes, max_edges