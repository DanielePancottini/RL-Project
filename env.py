#TODO Define the enviroment

import torch
import numpy as np

from gymnasium import Env, spaces

from torch_geometric.utils import subgraph

class Enviroment(Env):

    def __init__(self, graph, model, max_steps = 10):
        super(Enviroment, self).__init__()
        self.graph = graph
        self.model = model
        self.num_nodes = self.graph.x.shape[0]
        self.features_dim = self.graph.x.shape[1]
        self.max_steps = max_steps
        self.current_step = 0
        self.subgraph_mask = torch.zeros(self.num_nodes, dtype=bool)
        self.graph_attention = torch.zeros(self.num_nodes)

        #Compute attention for each node in the original graph
        self.graph_attention = self._compute_attention(self.graph.x, self.graph.edge_index)

        #Observation space
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-np.inf, np.inf, shape=self.graph.x.shape, dtype=np.float32),
            "edge_index": spaces.Box(0, self.num_nodes - 1, shape=self.graph.edge_index.shape, dtype=np.int64),
        })

        #Action space
        self.action_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (self.num_nodes, ),
            dtype = np.float32
        )

    """
        Execute the input action into the enviroment
    """
    def step(self, action):

        selected_nodes = torch.tensor([prob > np.random.uniform(0, 1) for prob in action])
        self.subgraph_mask = self.subgraph_mask | selected_nodes

        # Calculate reward based on the current subgraph's interpretability or performance
        reward = self._calculate_reward()

        #Check termination
        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}

    """
        Reset the enviroment
    """
    def reset(self):
        self.current_step = 0
        self.subgraph_mask = torch.zeros(self.num_nodes)
        self.graph_attention = torch.zeros(self.num_nodes)
        return self._get_observation()
        
    def _calculate_reward(self):

        #Compute the attention of the subgraph
        sub_edge_index = subgraph(self.subgraph_mask, self.graph.edge_index)
        sub_attention = self._compute_attention(self.graph.x[self.subgraph_mask], sub_edge_index)

        # Attention alignment reward
        attention_reward = sub_attention.sum() / self.graph_attention.sum()

        # Task-specific reward (e.g., quality of the subgraph)
        subgraph_nodes = np.where(self.subgraph_mask > 0)[0]
        task_reward = self._evaluate_subgraph(subgraph_nodes)

        # Composite reward
        alpha, beta = 0.5, 0.5  # Tune weights
        return alpha * task_reward + beta * attention_reward

    def _evaluate_subgraph(self, subgraph_nodes):
        # Placeholder for a task-specific evaluation, e.g., classification accuracy
        # Replace with your task's logic
        return len(subgraph_nodes) / self.num_nodes

    def _get_observation(self):
        return {
            "features": self.graph.x[self.subgraph_mask].numpy(),
            "edge_index": subgraph(self.subgraph_mask, self.graph.edge_index).numpy()
        }

    def _compute_attention(self, features, edge_index):
        """
        Recompute node-level attention weights using the GAT model.

        Returns:
            np.ndarray: Updated node-level attention weights for the selected subgraph.
        """
        self.model.eval()
        with torch.no_grad():
            _, (edge_index, attention) = self.model(features, edge_index)
            node_attention = torch.zeros(self.num_nodes)  # Initialize node attention scores

            # Aggregate attention scores for incoming edges
            for (src, dest), score in zip(edge_index.t().tolist(), attention.tolist()):
                node_attention[dest] += score

        return node_attention.numpy()
