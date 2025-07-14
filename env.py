import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

class GNNInterpretEnvironment(gym.Env):
    def __init__(self, gnn_model, single_graph_dataloader, max_nodes, max_edges, node_feature_dim, device='cuda', max_steps_per_episode=50):
        super().__init__()
        self.gnn_model = gnn_model.to(device)
        self.dataloader = single_graph_dataloader
        self.data_iter = iter(self.dataloader)
        self.device = device
        
        # Get max nodes/edges across over the possible batches
        # We can define fixed action and obeservation spaces

        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_features_dim = node_feature_dim
        self.max_steps_per_episode = max_steps_per_episode
        
        # Action space: A discrete choice to select an edge to "cut" (mask out), or a "stop" action.
        # Edge indices range from 0 to (self.max_edges - 1). The index self.max_edges represents "stop".
        self.action_space = spaces.Discrete(self.max_edges + 1) # +1 for the 'stop' action

        # Observation space: Fixed-size padded components representing the current graph state.
        self.observation_space = spaces.Dict({
            'x': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.max_nodes, self.node_features_dim), 
                dtype=np.float32
            ),
            'edge_index': spaces.Box(
                low=0, high=self.max_nodes - 1, # Node indices for edge_index can't exceed max_nodes - 1
                shape=(2, self.max_edges), 
                dtype=np.int64
            ),
            'current_edge_mask': spaces.Box( # The current binary mask applied to edges (0=cut, 1=present)
                low=0.0, high=1.0, 
                shape=(self.max_edges,), 
                dtype=np.float32
            ),
            'num_nodes': gym.spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int64),
            'num_edges': gym.spaces.Box(low=0, high=self.max_edges, shape=(1,), dtype=np.int64),
            'steps_taken': spaces.Box(low=0, high=self.max_steps_per_episode, shape=(1,), dtype=np.int32) # Current step count
        })

        # Additional parameters
        self.current_graph = None # Stores the PyG Data object for the active episode
        self.original_prediction_logits = None # Store the baseline model's prediction for fidelity
        self.current_edge_mask = None # The dynamic edge mask that the agent modifies
        self.num_steps_taken = 0 # Tracks steps within the current episode

    def _get_obs(self):
        """Helper function to construct the observation dictionary from the current graph state."""
        # Pad node features to max_nodes
        x_padded = torch.zeros((self.max_nodes, self.node_features_dim), 
                               dtype=self.current_graph.x.dtype, device=self.device)
        x_padded[:self.current_graph.x.size(0)] = self.current_graph.x

        # Pad edge_index to max_edges
        edge_index_padded = torch.zeros((2, self.max_edges), 
                                        dtype=torch.int64, device=self.device)
        if self.current_graph.edge_index.numel() > 0:
            edge_index_padded[:, :self.current_graph.edge_index.size(1)] = self.current_graph.edge_index
        
        # Pad the *current* edge mask to max_edges for consistent observation shape
        current_edge_mask_padded = np.pad(self.current_edge_mask.cpu().numpy(), 
                                          (0, self.max_edges - self.current_edge_mask.numel()), 
                                          'constant', constant_values=1.0) # Pad with 1s (unmasked/present)

        print(f"Env DEBUG: num_nodes type: {type(self.current_graph.num_nodes)}, value: {self.current_graph.num_nodes}")

        return {
            'x': x_padded.cpu().numpy(),
            'edge_index': edge_index_padded.cpu().numpy(),
            'current_edge_mask': current_edge_mask_padded.astype(np.float32),
            'num_nodes': np.array([self.current_graph.num_nodes], dtype=np.int64),
            'num_edges': np.array([self.current_graph.num_edges], dtype=np.int64),
            'steps_taken': np.array([self.num_steps_taken], dtype=np.int32)
        }
    
    
    def reset(self, seed=None, options=None):
        """Resets the environment, loading a new graph for a new interpretation episode."""
        super().reset(seed=seed)
        
        try:
            # Fetch a single graph from the dataloader (which must have batch_size=1)
            batch_data = next(self.data_iter)
            # Dataloader with batch_size=1 yields a Batch object with one graph, so extract it
            if isinstance(batch_data, Batch):
                self.current_graph = batch_data[0].to(self.device)
            else: # Fallback if dataloader directly yields Data objects
                self.current_graph = batch_data.to(self.device)
        except StopIteration:
            print("GNNInterpretEnvironment: DataLoader exhausted. Resetting iterator for a new epoch.")
            self.data_iter = iter(self.dataloader) # Restart the iterator for a new epoch
            batch_data = next(self.data_iter)
            if isinstance(batch_data, Batch):
                self.current_graph = batch_data[0].to(self.device)
            else:
                self.current_graph = batch_data.to(self.device)
        
        # Initialize the edge mask for the newly loaded graph to all ones (all edges are initially present)
        self.current_edge_mask = torch.ones(self.current_graph.num_edges, 
                                            dtype=torch.float32, device=self.device)
        self.num_steps_taken = 0 # Reset step counter for the new episode

        # Get the original GNN prediction for this new graph (for fidelity reward)
        with torch.no_grad():
            self.gnn_model.eval()
            self.original_prediction_logits = self.gnn_model(self.current_graph)
            
        observation = self._get_obs() # Get initial observation

         # Add this debug print
        print(f"Reset DEBUG: num_nodes from _get_obs() in reset(): type={type(observation['num_nodes'])}, value={observation['num_nodes']}")
        print(f"Reset DEBUG: num_nodes shape after _get_obs() in reset(): {observation['num_nodes'].shape if isinstance(observation['num_nodes'], np.ndarray) else 'Not a numpy array'}")


        info = {
            'original_graph_num_nodes': self.current_graph.num_nodes,
            'original_graph_num_edges': self.current_graph.num_edges,
            'original_prediction_class': self.original_prediction_logits.argmax(dim=1).item()
        }

        return observation, info
    
    def step(self, action):
        """
        Executes one step in the environment based on the agent's action.
        The action is a discrete choice: an edge index to cut, or a 'stop' signal.
        """
        terminated = False
        truncated = False 
        reward = 0.0

        chosen_edge_idx = action # The agent's action (integer)
        
        # Apply the chosen action
        if chosen_edge_idx < self.current_graph.num_edges: # Agent chose to cut a valid edge
            if self.current_edge_mask[chosen_edge_idx] == 1.0: # If the edge is currently present
                self.current_edge_mask[chosen_edge_idx] = 0.0 # Mark it as "cut" (masked out)
                # You might add a small reward or penalty here based on individual cut
                # E.g., reward -= 0.001 for each cut to encourage fewer cuts, but balance with fidelity
            else:
                # Penalty for choosing an edge that's already cut (inefficient action)
                reward -= 0.05 
        elif chosen_edge_idx == self.max_edges: # Agent chose the 'stop' action
            terminated = True # Episode ends
        else: # Invalid action (e.g., chosen_edge_idx > current_graph.num_edges but not 'stop')
            reward -= 0.1 # Penalty for invalid action
            
        self.num_steps_taken += 1 # Increment step counter

        # --- Calculate Reward for the current state of the masked graph ---
        # 1. Construct the masked graph based on the current_edge_mask
        # Filter edges based on the current mask
        masked_edge_indices_to_keep = (self.current_edge_mask > 0.5).nonzero(as_tuple=True)[0]
        
        masked_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device) # Initialize as empty
        if masked_edge_indices_to_keep.numel() > 0: # Only if at least one edge is kept
            masked_edge_index = self.current_graph.edge_index[:, masked_edge_indices_to_keep]
        
        # Create a PyG Data object for the masked graph. Node features are NOT masked here.
        masked_graph_data = Data(
            x=self.current_graph.x, 
            edge_index=masked_edge_index,
            y=self.current_graph.y, # Keep original label
            num_nodes=self.current_graph.num_nodes
        ).to(self.device)
        
        # Get prediction from the GNN for the currently masked graph
        with torch.no_grad():
            self.gnn_model.eval()
            masked_pred_logits = self.gnn_model(masked_graph_data)

        # 2. Fidelity Reward: How similar is the masked prediction to the original?
        original_target_class = self.original_prediction_logits.argmax(dim=1).long()
        fidelity_reward = -F.cross_entropy(masked_pred_logits, original_target_class, reduction='mean').item() # .item() for scalar

        # 3. Sparsity Reward: Encourage fewer remaining edges (more edges cut)
        num_remaining_edges = self.current_edge_mask.sum().item()
        # Reward is proportional to the fraction of edges *removed*
        sparsity_reward_value = 0.0
        if self.current_graph.num_edges > 0:
            sparsity_reward_value = (self.current_graph.num_edges - num_remaining_edges) / self.current_graph.num_edges
        
        sparsity_reward_value *= 0.1 # Weight sparsity (tune this hyperparameter)

        # Combine rewards
        reward += fidelity_reward + sparsity_reward_value 
        
        # --- Episode Termination Conditions ---
        if self.num_steps_taken >= self.max_steps_per_episode:
            truncated = True # Terminate if max steps reached
        
        if terminated or truncated:
            # If episode ends, the next observation will be from a *new* graph (after reset)
            next_observation, next_info = self.reset() # Prepare for next episode
            info_for_this_step = {
                'fidelity': fidelity_reward,
                'sparsity_metric': sparsity_reward_value, # Store the actual sparsity value
                'num_remaining_edges': num_remaining_edges,
                'steps_taken_in_episode': self.num_steps_taken,
                **next_info # Include info from the subsequent reset for context
            }
        else:
            # If episode continues, the observation reflects the current state of *this* graph
            next_observation = self._get_obs()
            info_for_this_step = {
                'fidelity': fidelity_reward,
                'sparsity_metric': sparsity_reward_value,
                'num_remaining_edges': num_remaining_edges,
                'steps_taken_in_episode': self.num_steps_taken
            }

        return next_observation, reward, terminated or truncated, False, info_for_this_step