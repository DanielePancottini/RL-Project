import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, degree
import networkx as nx

class GNNInterpretEnvironment(gym.Env):
    def __init__(self, gnn_model, single_graph_dataloader, max_nodes,
        max_edges, node_features_dim, max_explanation_nodes=50, device='cuda', max_steps_per_episode=50,
        fidelity_weight=1.0, sparsity_weight=1.0, invalid_action_penalty=-1.0,
        step_penalty=-0.01, stop_action_reward=1.0):
        super(GNNInterpretEnvironment, self).__init__()

        self.gnn_model = gnn_model.to(device)
        self.dataloader = single_graph_dataloader
        self.data_iter = iter(self.dataloader)
        self.device = device
        
        self.max_steps_per_episode = max_steps_per_episode
        self.max_nodes = max_nodes
        self.original_max_edges_for_padding = max_edges
        self.node_features_dim = node_features_dim
        self.max_explanation_nodes = max_explanation_nodes

        # Reward weights
        self.fidelity_weight = fidelity_weight
        self.sparsity_weight = sparsity_weight # Renamed for clarity and consistency
        self.invalid_action_penalty = invalid_action_penalty # Penalty for cutting already cut edge or invalid index
        self.step_penalty = step_penalty # Small penalty per step to encourage efficiency
        self.stop_action_reward = stop_action_reward # Reward for explicitly choosing to stop
        self.radius_penalty_weight = 0.5
        self.similarity_loss_weight = 0.5

        self.current_graph = None
        self.original_prediction_logits = None
        self.original_target_class = None
        self.num_steps_taken = 0
        self.current_explanation_nodes = set()
        self.initial_node_idx = None # To store the seed node index

        # Define observation space
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes, self.node_features_dim), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=self.max_nodes - 1, shape=(2, self.original_max_edges_for_padding), dtype=np.int64),
            'node_mask': spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes,), dtype=np.float32),
            'valid_action_mask': spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes + 1,), dtype=np.float32),
            'steps_taken': spaces.Box(low=0, high=self.max_steps_per_episode, shape=(1,), dtype=np.int32),
            'num_nodes': spaces.Box(low=0, high=self.max_nodes, shape=(1,), dtype=np.int32),
            'num_edges': spaces.Box(low=0, high=self.original_max_edges_for_padding, shape=(1,), dtype=np.int32),
            'is_start_node_mask': spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes,), dtype=np.float32)
        })

        # Define action space (node indices + a stop action)
        self.action_space = spaces.Discrete(self.max_nodes + 1)



    def _get_obs(self):
        # Pad node features 
        x_padded = torch.zeros((self.max_nodes, self.node_features_dim), 
                               dtype=self.current_graph.x.dtype, device=self.device)
        x_padded[:self.current_graph.x.size(0)] = self.current_graph.x

        # Pad edge_index
        edge_index_padded = torch.zeros((2, self.original_max_edges_for_padding), dtype=torch.long, device=self.device)
        num_edges_to_copy = min(self.current_graph.edge_index.size(1), self.original_max_edges_for_padding)
        edge_index_padded[:, :num_edges_to_copy] = self.current_graph.edge_index
        
        node_mask = torch.zeros(self.max_nodes, dtype=torch.float32, device=self.device)
        for node_idx in self.current_explanation_nodes:
            if node_idx < self.max_nodes: # Ensure node index is within padded range
                node_mask[node_idx] = 1.0
        
        # Create is_start_node_mask
        is_start_node_mask = torch.zeros(self.max_nodes, dtype=torch.float32, device=self.device)
        if self.initial_node_idx is not None and self.initial_node_idx < self.current_graph.num_nodes:
            is_start_node_mask[self.initial_node_idx] = 1.0

        # Create valid action mask
        valid_action_mask = torch.zeros(self.max_nodes + 1, dtype=torch.float32, device=self.device)
        
        # Determine valid next nodes to add (1-hop neighbors not already in explanation)
        if self.current_graph.edge_index.numel() > 0:
            adj_list = [[] for _ in range(self.current_graph.num_nodes)]
            for src, dest in self.current_graph.edge_index.t().tolist():
                adj_list[src].append(dest)
                adj_list[dest].append(src) # Assuming undirected for simplicity or based on graph type

            # Find neighbors of nodes currently in explanation
            neighbors_of_explanation = set()
            for exp_node in self.current_explanation_nodes:
                if exp_node < self.current_graph.num_nodes:
                    for neighbor in adj_list[exp_node]:
                        if neighbor not in self.current_explanation_nodes and neighbor < self.max_nodes:
                            neighbors_of_explanation.add(neighbor)
            
            for node_idx in neighbors_of_explanation:
                valid_action_mask[node_idx] = 1.0

        # The STOP action is always valid
        valid_action_mask[self.max_nodes] = 1.0

        return {
            'x': x_padded.cpu().numpy(),
            'edge_index': edge_index_padded.cpu().numpy(),
            'node_mask': node_mask.cpu().numpy(),
            'valid_action_mask': valid_action_mask.cpu().numpy(),
            'steps_taken': np.array([self.num_steps_taken], dtype=np.int32),
            'num_nodes': np.array([self.current_graph.num_nodes], dtype=np.int32),
            'num_edges': np.array([self.current_graph.edge_index.size(1)], dtype=np.int32),
            'is_start_node_mask': is_start_node_mask.cpu().numpy()
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            current_graph_batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            current_graph_batch  = next(self.data_iter)
            
        # Handle both Batch and Data objects
        if isinstance(current_graph_batch, Batch):
            if current_graph_batch.num_graphs > 1:
                self.current_graph = current_graph_batch.get_example(0).to(self.device)
            else:
                self.current_graph = current_graph_batch.to(self.device)
        else:
            self.current_graph = current_graph_batch.to(self.device)
        
        # Initial node selection (using heuristic)
        initial_node = None
        if self.current_graph.edge_index.numel() > 0: # Ensure graph has edges for degree calculation
            node_degrees = degree(self.current_graph.edge_index[0], num_nodes=self.current_graph.num_nodes)
            initial_node = torch.argmax(node_degrees).item()
        else: # Fallback for graphs with no edges or for safety
            initial_node = 0
        
        self.initial_node_idx = initial_node # Store the initial node index

        # Reset explanation state
        self.num_steps_taken = 0
        self.current_explanation_nodes = {self.initial_node_idx}

        # Get original prediction of the GNN model
        self.gnn_model.eval() # Set GNN to evaluation mode
        with torch.no_grad():
            if isinstance(self.current_graph, Batch): # Handle batch case if model expects it
                original_pred_logits = self.gnn_model(self.current_graph)
                # If model returns batch logits, take for the first graph
                if original_pred_logits.dim() > 1 and original_pred_logits.size(0) > 1:
                    original_pred_logits = original_pred_logits[0].unsqueeze(0)
            else:
                original_pred_logits = self.gnn_model(self.current_graph)
            
            self.original_prediction_logits = original_pred_logits.squeeze(0) # Store as 1D tensor

            self.original_target_class = torch.argmax(self.original_prediction_logits).item()

        self.original_graph_num_nodes = self.current_graph.num_nodes
        self.original_graph_num_edges = self.current_graph.edge_index.size(1)

        observation = self._get_obs()
        info = {}

        return observation, info
    
    def calculate_metrics(self):
        num_explanation_nodes = len(self.current_explanation_nodes)

        # Handle empty explanation
        if not self.current_explanation_nodes:
            fidelity_score = 0.0 # No prediction on empty subgraph
            sparsity_score = 1.0 # Max sparsity
            radius_penalty_score = 0.0 # No penalty for empty
            similarity_loss_score = 0.0 # No loss for empty
            return fidelity_score, sparsity_score, radius_penalty_score, similarity_loss_score, num_explanation_nodes

        nodes_in_exp_tensor = torch.tensor(list(self.current_explanation_nodes), dtype=torch.long, device=self.device)

        # Create subgraph
        # Note: subgraph automatically relabels nodes by default, but it's crucial
        # that the GNN model (or feature extractor) correctly handles this.
        # The 'relabel_nodes=True' means node features 'x' for the subgraph will
        # only contain features of the selected nodes, indexed from 0 to num_explanation_nodes-1.
        # The 'edge_index' will also be re-indexed.
        sub_edge_index, _ = subgraph(
            nodes_in_exp_tensor,
            self.current_graph.edge_index,
            relabel_nodes=True,
            num_nodes=self.current_graph.num_nodes, 
            return_edge_mask=False
        )

        sub_x = self.current_graph.x[nodes_in_exp_tensor]

        # Create a new Data object for the masked subgraph
        masked_graph_data = Data(x=sub_x, edge_index=sub_edge_index)
        if hasattr(self.current_graph, 'batch'): # For batched graphs from dataloader
            masked_graph_data.batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=self.device) # Single graph in batch

        # Get prediction on the masked subgraph
        self.gnn_model.eval() # Set GNN to evaluation mode
        with torch.no_grad():
            masked_pred_logits = self.gnn_model(masked_graph_data).squeeze(0)

        # Fidelity calculation
        masked_pred_probs = F.softmax(masked_pred_logits, dim=-1)
        fidelity_score = masked_pred_probs[self.original_target_class].item()

        # Sparsity calculation: (Total Nodes - Explanation Nodes) / Total Nodes
        # Higher is better for sparsity.
        sparsity_score = (self.original_graph_num_nodes - num_explanation_nodes) / self.original_graph_num_nodes

        # Radius Penalty: max shortest path distance from seed node (v0) to any node in S
        radius_penalty_score = 0.0
        if self.initial_node_idx is not None and num_explanation_nodes > 0:
            if self.current_graph.edge_index.numel() > 0 and self.current_graph.num_nodes > 0:
                g_nx = nx.Graph()
                g_nx.add_nodes_from(range(self.current_graph.num_nodes))
                g_nx.add_edges_from(self.current_graph.edge_index.t().tolist())

                if self.initial_node_idx in g_nx:
                    max_distance = 0
                    # Only consider nodes that are actually in the explanation
                    for node_in_exp in self.current_explanation_nodes:
                        if node_in_exp in g_nx and nx.has_path(g_nx, self.initial_node_idx, node_in_exp):
                            distance = nx.shortest_path_length(g_nx, source=self.initial_node_idx, target=node_in_exp)
                            if distance > max_distance:
                                max_distance = distance
                    radius_penalty_score = float(max_distance)
                else:
                    # Initial node not in graph (e.g., graph is empty or initial node out of bounds)
                    # Assign a default penalty or 0
                    radius_penalty_score = 0.0
            else:
                # Graph has no edges or nodes, distances are 0
                radius_penalty_score = 0.0

        # Similarity Loss: ||H̄T(v0) - zv0||2
        # This requires node embeddings. Your GNN model (GNNWithAttention) currently outputs
        # graph-level predictions. You would need to modify GNNWithAttention to return
        # intermediate node embeddings and ensure GNNFeatureExtractor also passes them.
        # This is a significant architectural change to the baseline GNN.
        # For now, this is a placeholder. Without it, the paper's full reward is not implemented.
        similarity_loss_score = 0.0 
        # To implement:
        # 1. Modify GNNWithAttention to return original node embeddings (zv0 for initial_node_idx).
        # 2. Modify GNNWithAttention to return subgraph node embeddings (H̄T(v0) for initial_node_idx).
        # 3. Calculate F.mse_loss(H̄T(v0), zv0)

        return fidelity_score, sparsity_score, radius_penalty_score, similarity_loss_score, num_explanation_nodes

    def step(self, action):
        terminated = False
        truncated = False 
        reward = self.step_penalty # Base penalty for each step

        self.num_steps_taken += 1 # Increment step counter

        chosen_node_or_stop_action = action
        current_valid_action_mask = self._get_obs()['valid_action_mask'] # Get fresh mask

        # --- Process Action ---
        if chosen_node_or_stop_action == self.max_nodes: # STOP action
            terminated = True
            # The stop_action_reward is part of the final episodic reward if the agent chooses to stop
        else: # Node adding action
            node_to_add = chosen_node_or_stop_action
            is_valid_action = (current_valid_action_mask[node_to_add] == 1.0)

            if not is_valid_action:
                reward += self.invalid_action_penalty # Apply penalty immediately
            else:
                self.current_explanation_nodes.add(node_to_add)

        # --- Episode Termination Conditions ---
        if self.num_steps_taken >= self.max_steps_per_episode:
            truncated = True # Episode truncated due to max steps

        # --- Calculate Final Episodic Reward ---
        # The paper calculates the full reward (negative loss) only upon episode completion.
        if terminated or truncated:
            fidelity_score, sparsity_score, radius_penalty_score, similarity_loss_score, num_explanation_nodes = self.calculate_metrics()
            
            # The paper's L(S) combines these as *losses* to minimize. Reward is -L(S).
            # L(S) = Prediction Loss + λ1 * Size Loss + λ2 * Radius Penalty + λ3 * Similarity Loss
            # Your fidelity_score is probability (higher is better), so Prediction Loss is (1 - fidelity_score)
            # Your sparsity_score is (1 - num_nodes/total_nodes) (higher is better), so Size Loss is (1 - sparsity_score)
            
            # Ensure num_explanation_nodes is not 0 for size loss calculation to avoid div by zero if graph has 0 nodes
            current_size_loss = num_explanation_nodes / self.original_graph_num_nodes if self.original_graph_num_nodes > 0 else 0.0

            total_loss = (1.0 - fidelity_score) * self.fidelity_weight + \
                         current_size_loss * self.sparsity_weight + \
                         radius_penalty_score * self.radius_penalty_weight + \
                         similarity_loss_score * self.similarity_loss_weight
            
            # The total reward for the episode
            # If the agent chose to stop, provide the stop_action_reward in addition to the loss/penalty.
            final_episodic_reward = -total_loss
            if chosen_node_or_stop_action == self.max_nodes:
                 final_episodic_reward += self.stop_action_reward # This is a positive reward component
            
            reward += final_episodic_reward # Add to any step penalties

            # Prepare for next episode
            next_observation, next_info = self.reset()
            info_for_this_step = {
                'fidelity': fidelity_score,
                'sparsity_metric': sparsity_score,
                'radius_penalty': radius_penalty_score,
                'similarity_loss': similarity_loss_score,
                'num_explanation_nodes': num_explanation_nodes,
                'steps_taken_in_episode': self.num_steps_taken,
                'final_episode_loss': total_loss,
                **next_info # Include info from reset (e.g., initial graph details)
            }
        else: # Not terminated/truncated yet, so only step_penalty and invalid_action_penalty apply
            next_observation = self._get_obs()
            # Info during intermediate steps (can be empty or specific diagnostic)
            info_for_this_step = {
                'fidelity': 0.0, # Not meaningful during intermediate steps
                'sparsity_metric': 0.0, # Not meaningful during intermediate steps
                'radius_penalty': 0.0,
                'similarity_loss': 0.0,
                'num_explanation_nodes': len(self.current_explanation_nodes),
                'steps_taken_in_episode': self.num_steps_taken
            }

        return next_observation, reward, terminated, truncated, info_for_this_step