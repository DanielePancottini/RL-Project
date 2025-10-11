import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph, degree
import networkx as nx
from torch.distributions import Categorical
import collections
import random

class GNNInterpretEnvironment(gym.Env):
    def __init__(self, gnn_model, dataloader, max_steps = 50, device = 'cpu'):
        super().__init__()

        self.device = device
        self.gnn_model = gnn_model.to(self.device)
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)
        self.max_steps = max_steps

        # Reward weights
        self.size_weight = 0.01
        self.radius_penalty_weight = 0.01
        self.similarity_loss_weight = 0.01

        # Runtime state
        self.current_graph = None
        self.initial_node = None
        self.S = None
        self.steps_taken = 0
        self.adj_list = None
        self.dist_from_seed = {}
        self.original_pred_logits = None
        self.original_target_class = None
        self.original_seed_node_embedding = None

        # No gymnasium action and observation spaces defined because we need
        # dynamic spaces since each graph has its own dimensions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)

        if isinstance(data, Batch):
            graph = data.get_example(0).to(self.device)
        else:
            graph = data.to(self.device)

        self.current_graph = graph
        self.steps_taken = 0

        # choose seed: gradient-based
        self.initial_node = self._choose_seed_gradient(graph)
        
        # choose seed: random
        #self.initial_node = random.randint(0, graph.num_nodes - 1)

        self.S = {self.initial_node}
        self._precompute_graph_structures()

        # store original pred and seed embedding
        self.gnn_model.eval()
        with torch.no_grad():
            logits = self.gnn_model(self.current_graph.x, self.current_graph.edge_index)
            logits = logits.squeeze(0)
            self.original_pred_logits = logits.detach()
            self.original_target_class = int(torch.argmax(logits).item())
            self.original_seed_node_embedding = self.gnn_model.embedding(self.current_graph.x, self.current_graph.edge_index)[self.initial_node].detach()

        return self._make_obs(self.S), {}

    def _choose_seed_gradient(self, graph):
        graph.x.requires_grad = True
        self.gnn_model.eval()

        logits = self.gnn_model(graph.x, graph.edge_index).squeeze(0)
        target_class = logits.argmax()
        
        self.gnn_model.zero_grad()
        logits[target_class].backward()
        
        # Node importance = L2 norm of gradients
        node_importance = graph.x.grad.norm(dim=1)
        seed_node = int(torch.argmax(node_importance).item())
        
        graph.x.requires_grad = False
        return seed_node

    def _precompute_graph_structures(self):
        
        self.adj_list = [[] for _ in range(self.current_graph.num_nodes)]
        if self.current_graph.edge_index.numel() > 0:
            for src, dst in self.current_graph.edge_index.t().tolist():
                self.adj_list[src].append(dst)
                self.adj_list[dst].append(src)
        
        # distances from seed
        try:
            g_nx = nx.Graph()
            g_nx.add_nodes_from(range(self.current_graph.num_nodes))
            g_nx.add_edges_from(self.current_graph.edge_index.t().tolist())
            if self.initial_node in g_nx:
                self.dist_from_seed = nx.single_source_shortest_path_length(g_nx, self.initial_node)
            else:
                self.dist_from_seed = {}
        except Exception:
            self.dist_from_seed = {}

    def _make_obs(self, S_local):
        
        S = S_local
        frontier = set()
        for u in S:
            for v in self.adj_list[u]:
                if v not in S:
                    frontier.add(v)
        frontier = sorted(frontier)

        nodes_local = sorted(list(S.union(frontier)))
        node_map = {g: i for i, g in enumerate(nodes_local)}
        n_local = len(nodes_local)

        # Augmenting nodes features
        if n_local > 0:
            x_full = self.current_graph.x[nodes_local].to(self.device)  # (n_local, feat)
            is_start = torch.zeros(n_local, 1, device=self.device)
            is_start[node_map[self.initial_node], 0] = 1.0
            in_S_col = torch.tensor([[1.0 if g in S else 0.0] for g in nodes_local], device=self.device)
            x_aug = torch.cat([x_full, is_start, in_S_col], dim=-1)
        else:
            x_aug = torch.zeros((0, self.current_graph.x.size(1) + 2), device=self.device)

        # Normalize A matrix
        A = torch.zeros((n_local, n_local), dtype=torch.float32, device=self.device)
        for g in nodes_local:
            for nb in self.adj_list[g]:
                if nb in node_map:
                    A[node_map[g], node_map[nb]] = 1.0
        if n_local > 0:
            A = A + torch.eye(n_local, device=self.device)
            deg = A.sum(dim=1, keepdim=True)
            deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
            A_norm = deg_inv_sqrt * A * deg_inv_sqrt.t()
        else:
            A_norm = torch.zeros((0, 0), device=self.device)

        candidate_nodes = [node_map[g] for g in frontier]
        local_to_global = nodes_local

        obs = {
            'x_aug': x_aug,                   # (n_local, feat+2)
            'A_norm': A_norm,                 # (n_local, n_local)
            'candidate_nodes': candidate_nodes,
            'local_to_global': local_to_global,
            'S_size': len(S)
        }
        return obs

    def step(self, action_local):
        self.steps_taken += 1
        obs = self._make_obs(self.S)
        candidate_nodes = obs['candidate_nodes']
        stop_index = len(candidate_nodes)

        if action_local == stop_index:
            reward, info = self._compute_final_reward(self.S)
            return None, float(reward), True, False, info

        if action_local < 0 or action_local > stop_index:
            # invalid: ignore
            pass
        else:
            chosen_local = candidate_nodes[action_local]
            chosen_global = obs['local_to_global'][chosen_local]
            self.S.add(int(chosen_global))

        if self.steps_taken >= self.max_steps:
            reward, info = self._compute_final_reward(self.S)
            return None, float(reward), False, True, info

        return self._make_obs(self.S), 0.0, False, False, {}

    def _compute_final_reward(self, S_local):
        if not S_local:
            return 0.0, {'fidelity': 0.0, 'sparsity_prop': 1.0, 'radius': 0.0, 'similarity': 0.0, 'size': 0}

        nodes_sorted = sorted(list(S_local))
        nodes_tensor = torch.tensor(nodes_sorted, dtype=torch.long, device=self.device)

        sub_edge_index, _ = subgraph(nodes_tensor, self.current_graph.edge_index, relabel_nodes=True,
                                     num_nodes=self.current_graph.num_nodes)
        sub_x = self.current_graph.x[nodes_tensor].to(self.device)
        sub_data = Data(x=sub_x, edge_index=sub_edge_index)
        sub_data.batch = torch.zeros(sub_x.size(0), dtype=torch.long, device=self.device)
        
        node_map = {g.item(): i for i, g in enumerate(nodes_tensor)}

        self.gnn_model.eval()
        with torch.no_grad():
            masked_logits = self.gnn_model(sub_data.x, sub_data.edge_index).squeeze(0)
            masked_seed_embedding = self.gnn_model.embedding(sub_data.x, sub_data.edge_index)[node_map[self.initial_node]].unsqueeze(0)

        #Prediction Loss
        original_probs = F.softmax(self.original_pred_logits, dim=-1)
        masked_log_probs = F.log_softmax(masked_logits, dim=-1)
        prediction = - (original_probs * masked_log_probs).sum()

        #Size Loss
        size_loss = len(S_local)
        
        #Radius Loss
        radius = 0.0
        for n in S_local:
            d = self.dist_from_seed.get(n, 0)
            if d > radius:
                radius = float(d)

        #Similarity Loss
        similarity = torch.norm(self.original_seed_node_embedding - masked_seed_embedding, p=2).item()

        total_loss = prediction + \
                     size_loss * self.size_weight + \
                     radius * self.radius_penalty_weight + \
                     similarity * self.similarity_loss_weight

        reward = -total_loss.detach()
        info = {'prediction': prediction, 'size loss': size_loss, 'radius': radius, 'similarity': similarity, 'size': len(S_local)}
        return reward, info

    def simulate_rollout_from_S(self, S_init, policy, max_rollout_steps = None, stochastic = True):
        if max_rollout_steps is None:
            max_rollout_steps = self.max_steps
        
        S_local = set(S_init)  # copy

        steps = 0
        while True:
            obs_local = self._make_obs(S_local)
            candidates = list(obs_local['candidate_nodes'])
            S_indices = [obs_local['local_to_global'].index(n) for n in S_local]
            stop_idx = len(candidates)
            with torch.no_grad():
                probs, _, _ = policy(obs_local['x_aug'], obs_local['A_norm'], candidates, S_indices)
            if probs.numel() == 1:
                act_idx = 0
            else:
                if stochastic:
                    dist = Categorical(probs)
                    act_idx = dist.sample().item()
                else:
                    act_idx = int(torch.argmax(probs).item())
            if act_idx == stop_idx:
                reward, info = self._compute_final_reward(S_local)
                return reward, info, S_local
            chosen_local = candidates[act_idx]
            chosen_global = obs_local['local_to_global'][chosen_local]
            S_local.add(int(chosen_global))
            steps += 1
            if steps >= max_rollout_steps:
                reward, info = self._compute_final_reward(S_local)
                return reward, info, S_local

    def estimate_Q(self, S_after_action: set, policy, M = 10, max_rollout_steps = None):
        rewards = []
        for i in range(M):
            r, _, _ = self.simulate_rollout_from_S(S_after_action, policy, max_rollout_steps=max_rollout_steps, stochastic=True)
            rewards.append(r)
        return torch.stack(rewards).mean().item(), rewards