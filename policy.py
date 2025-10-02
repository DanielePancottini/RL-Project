import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import degree

from layers import SelfAttnPooling, Swish

class Policy(nn.Module):
    """
    Policy implementing Θ1 + APPNP-like propagation + MLP + scorer θ3.
    Outputs a probability distribution over candidate nodes + STOP.
    """
    def __init__(self, input_dim, hidden_dim = 128, L = 3, alpha = 0.85, device = 'cpu') -> None:
        super().__init__()

        self.device = device
        self.L = L
        self.alpha = alpha
        self.theta1 = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_score = nn.Linear(hidden_dim, 1, bias=False)
        self.stop_layer = nn.Linear(hidden_dim, 2, bias=False)
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.pooling = nn.Sequential(SelfAttnPooling(hidden_dim), Swish())

        nn.init.zeros_(self.value_layer.weight.data)
        nn.init.zeros_(self.node_score.weight.data)
        nn.init.zeros_(self.stop_layer.weight.data)


    def propagate(self, x, A):
        """APPNP-like propagation"""
        H0 = self.theta1(x)
        H = H0.clone()
        for _ in range(self.L):
            H = (1.0 - self.alpha) * (A @ H) + self.alpha * H0
        H_bar = self.mlp(H)
        return H_bar

    def forward(self, x_aug, A_norm, candidate_nodes, S_indices, H_global):

        H_delta = self.propagate(x_aug, A_norm)
        H_global[S_indices] += H_delta[S_indices]  # accumulate embeddings
        global_z = self.pooling(H_global[S_indices])

        if len(candidate_nodes) == 0:
            logits_candidates = torch.tensor([], device=self.device)
        else:
            idx = torch.tensor(candidate_nodes, dtype=torch.long, device=self.device)
            cand_emb = H_global[idx]    
            node_scores = self.node_score(cand_emb).squeeze(-1)      # (num_candidates,)
            logits_candidates = F.log_softmax(node_scores, dim=0) 

        stop_logits = torch.log_softmax(self.stop_layer(global_z), dim=0)
        all_logits = torch.cat([logits_candidates + stop_logits[0], stop_logits[1:]], dim=0)
        value = self.value_layer(global_z).squeeze()
        return all_logits, value, H_global

"""
    Pretrain Policy
"""
def pretrain_policy(policy, optimizer, expert_trajectories, env, epochs=25, device='cpu'):
    """
    Pre-trains the policy using imitation learning on expert trajectories.
    This corresponds to the paper's `pretrain_list` phase.
    """
    print("Starting Policy Pre-training...")
    policy.to(device)
    policy.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        
        # Shuffle the trajectories for each epoch
        random.shuffle(expert_trajectories)
        
        for graph, expert_actions in tqdm(expert_trajectories, desc=f"Pre-train Epoch {epoch}/{epochs}"):
            
            # Manually reset the environment with the current graph
            # Note: This is a simplified reset, you may need to adapt it if your
            # env.reset() does more complex things.
            env.current_graph = graph.to(env.device)
            if graph.edge_index.numel() > 0:
                deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
                env.initial_node = int(torch.argmax(deg).item())
            else:
                env.initial_node = 0
            
            env.S = {env.initial_node}
            env._precompute_graph_structures()
            
            # Initialize hidden state for the policy
            H_global = torch.zeros((env.current_graph.num_nodes, policy.theta1.out_features), device=env.device)
            
            trajectory_loss = 0
            
            # Step through the expert trajectory
            for expert_node_to_add in expert_actions:
                # 1. Get the current state (observation) from the environment
                obs = env._make_obs(env.S)
                
                # 2. Get the policy's action probabilities (logits) for the current state
                logits, _, H_global_next = policy(obs['x_aug'], obs['A_norm'], obs['candidate_nodes'], 
                                                  [obs['local_to_global'].index(n) for n in env.S], H_global)
                H_global = H_global_next

                # 3. Find the index of the expert's action in the current candidate list
                # This is our target label for the supervised loss.
                candidate_nodes_global = [obs['local_to_global'][i] for i in obs['candidate_nodes']]
                
                try:
                    # The index of the expert action becomes our target class
                    target_action_index = candidate_nodes_global.index(expert_node_to_add)
                    target = torch.tensor([target_action_index], device=env.device)

                    # 4. Calculate the Cross-Entropy loss
                    # We need to ignore the final "STOP" action logit for this.
                    action_logits = logits[:-1] 
                    trajectory_loss += loss_fn(action_logits.unsqueeze(0), target)

                except ValueError:
                    # The expert node is not in the candidate list, which can happen. We just skip this step.
                    pass
                
                # 5. Manually apply the expert action to update the environment state for the next step
                env.S.add(expert_node_to_add)

            if trajectory_loss > 0:
                optimizer.zero_grad()
                trajectory_loss.backward()
                optimizer.step()
                total_loss += trajectory_loss.item()
        
        avg_loss = total_loss / len(expert_trajectories) if len(expert_trajectories) > 0 else 0
        print(f"[Pre-train Epoch {epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

"""
    Training Loop
"""

def train_reinforce_rollout(env,
                            policy,
                            optimizer,
                            episodes = 1000,
                            rollout_M = 10,
                            entropy_coeff = 1e-6,
                            rollout_max_steps = None,
                            baseline = True,
                            device = 'cpu',
                            log_every = 10):
    policy.to(device)
    for ep in range(1, episodes + 1):
    
        obs, _ = env.reset()
        traj_log_probs = []
        traj_Qs = []
        traj_actions = []
        traj_values = []
        traj_infos = []
        entropies = []
        done = False
        steps = 0

        H_global = torch.zeros((env.current_graph.num_nodes, policy.theta1.out_features), device=device) # Num nodes X Hidden Dim

        while not done:
            x_aug = obs['x_aug']
            A_norm = obs['A_norm']
            candidates = list(obs['candidate_nodes'])
            S_indices = [obs['local_to_global'].index(n) for n in env.S]

            logits, value, H_global = policy(x_aug, A_norm, candidates, S_indices, H_global)
            probs = torch.exp(logits)
            dist = Categorical(probs)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx)
            entropies.append(dist.entropy())

            # apply action on main env
            next_obs, terminal_reward, terminated, truncated, info = env.step(action_idx.item())

            # estimate Q:
            if terminated or truncated:
                q_est = torch.tensor(terminal_reward, dtype=torch.float32, device=device)  # true final reward
            else:
                # build S_after_action from next_obs: nodes with in_S==1 (last column)
                # next_obs['x_aug'] is tensor (n_local, feat+2); last column is in_S flag
                in_S_flags = next_obs['x_aug'][:, -1]  # tensor
                nodes_local = next_obs['local_to_global']
                S_after_action = set()
                for local_idx, flag in enumerate(in_S_flags):
                    if float(flag.item()) == 1.0:
                        S_after_action.add(int(nodes_local[local_idx]))
                q_est, rollout_rewards = env.estimate_Q(S_after_action, policy, H_global, M=rollout_M, max_rollout_steps=rollout_max_steps)
                q_est = torch.tensor(q_est, dtype=torch.float32, device=device)

            traj_log_probs.append(logp)
            traj_Qs.append(q_est)
            traj_actions.append(action_idx.item())
            traj_values.append(value.squeeze())
            traj_infos.append(info if terminated or truncated else {})
            obs = next_obs
            done = terminated or truncated
            steps += 1

        # Optionally use baseline: subtract mean Q across trajectory
        Qs = torch.stack(traj_Qs)
        values = torch.stack(traj_values)
        logps = torch.stack(traj_log_probs)

        if baseline:
            advantages = Qs - values.detach()
        else:
            advantages = Qs

        # Advantages normalization
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy loss: - sum_t logπ(a_t|s_{t-1}) * Q_hat (we use advantages or raw Q per choice)
        # Policy loss (actor)
        policy_loss = -(advantages * logps).mean()

        # Value loss (critic)
        value_loss = 0.5 * (Qs - values).pow(2).mean()

        # Entropy regularization
        entropy_loss = torch.stack(entropies).mean()

        # Total loss
        loss = policy_loss - entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Logging ----
        if ep % log_every == 0 or ep == 1:
            mean_final_reward = Qs[-1].item()
            mean_q = Qs.mean().item()
            print(f"[Ep {ep}/{episodes}] steps={steps} mean_rollout_Q={mean_q:.4f} "
                  f"final_Q={mean_final_reward:.4f} loss={policy_loss.item():.4f}")

        # DEBUG: per-step diagnostics
        print(f"Ep {ep} step {steps}: |S|={len(env.S)} candidates={len(candidates)} "
            f"probs_sum={float(probs.sum().item()):.4f}"
            f"probs={probs.detach().cpu().numpy()}")

    print("Training complete.")

