import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import degree

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
        self.node_score = nn.Linear(hidden_dim, 1)
        self.stop_score = nn.Linear(hidden_dim, 1)
        self.value_layer = nn.Linear(hidden_dim, 1)


    def propagate(self, x, A):
        """APPNP-like propagation"""
        H0 = self.theta1(x)
        H = H0.clone()
        for _ in range(self.L):
            H = (1.0 - self.alpha) * (A @ H) + self.alpha * H0
        H_bar = self.mlp(H)
        return H_bar

    def forward(self, x_aug, A_norm, candidate_nodes, S_indices):

        H_global = self.propagate(x_aug, A_norm)  # (num_nodes, hidden_dim)

        """Candidates score"""
        if len(candidate_nodes) == 0:
            candidate_scores = torch.tensor([], device=self.device)
        else:
            candidate_scores = self.node_score(H_global[candidate_nodes]).squeeze(-1)  # (num_candidates,)

        """STOP embedding and score"""
        S_tensor = torch.tensor(S_indices, dtype=torch.long, device=self.device) if len(S_indices) > 0 else torch.tensor([], dtype=torch.long, device=self.device)
        cand_tensor = torch.tensor(candidate_nodes, dtype=torch.long, device=self.device) if len(candidate_nodes) > 0 else torch.tensor([], dtype=torch.long, device=self.device)
        union_nodes = torch.cat([S_tensor, cand_tensor], dim=0) # Union = S_{t-1} ∪ ∂S_{t-1}

        union_emb = H_global[union_nodes]  # (|Union|, hidden_dim)

        # Attention for STOP
        attn_scores = self.stop_score(union_emb).squeeze(-1)  # (|Union|,)
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)  # (|Union|, 1)

        # Stop embedding
        H_stop = torch.sum(attn_weights * union_emb, dim=0)  # (hidden_dim,)

        #Stop score
        stop_score = self.node_score(H_stop).squeeze(-1)  # scalar

        """Final probabilities"""
        all_scores = torch.cat([candidate_scores, stop_score.unsqueeze(0)], dim=0)  # (num_candidates + 1,)
        all_probs = F.softmax(all_scores, dim=0)  # (num_candidates + 1,)

        """Value estimate"""
        value = self.value_layer(H_stop).squeeze(-1)  # scalar

        return all_probs, all_scores, value

"""
    Pretrain Policy
"""
def pretrain_policy_listwise(policy, optimizer, pretraining_samples, env, epochs=10, device='cpu'):
    """
    Pre-trains by dynamically generating a new random-length trajectory for each
    training instance, exactly as described in the paper's reference code.
    """
    print("Starting Policy Pre-training with dynamic trajectory generation...")
    policy.to(device)
    policy.train()

    # Use teacher-forcing pretraining as described
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"--- Pretrain Epoch {epoch+1}/{epochs} ---")
        random.shuffle(pretraining_samples)
        seqs_processed = 0
        total_loss_val = 0.0

        for graph, seed_node, bfs_sequence in tqdm(pretraining_samples, desc=f"Pretrain Epoch {epoch+1}"):
            # Initialize environment and graph structures for this sample
            env.current_graph = graph.to(device)
            env.initial_node = seed_node
            env._precompute_graph_structures()

            # Teacher forcing: start from the seed
            current_subgraph_nodes = {seed_node}
            accumulated_loss = 0

            max_len = len(bfs_sequence) - 1
            if max_len <= 0:
                continue
            trajectory_len = random.randint(1, max_len)

            # Iterate through sequence and predict next node at each step
            for t in range(trajectory_len):
                obs = env._make_obs(current_subgraph_nodes)

                # boundary as global node ids
                boundary_global = [obs['local_to_global'][i] for i in obs['candidate_nodes']]
                if not boundary_global:
                    break

                S_indices = [obs['local_to_global'].index(n) for n in sorted(list(current_subgraph_nodes))]

                # Raw scores (logits) for candidates + STOP
                _, all_scores, _ = policy(obs['x_aug'], obs['A_norm'], obs['candidate_nodes'], S_indices)
                action_logits = all_scores[:-1]

                if action_logits.numel() == 0:
                    break

                target_node = bfs_sequence[t+1]
                if target_node not in boundary_global:
                    # If target not in boundary, we cannot train this step
                    break

                target_node_index = boundary_global.index(target_node)
                target_tensor = torch.tensor([target_node_index], dtype=torch.long, device=device)

                step_loss = loss_fn(action_logits.unsqueeze(0), target_tensor)
                accumulated_loss = accumulated_loss + step_loss if isinstance(accumulated_loss, torch.Tensor) else (step_loss + accumulated_loss)

                # Teacher forcing: add correct node for next step
                current_subgraph_nodes.add(target_node)

            # Backpropagate after the full sequence
            if isinstance(accumulated_loss, torch.Tensor) and accumulated_loss.item() > 0:
                optimizer.zero_grad()
                accumulated_loss.backward()
                optimizer.step()
                seqs_processed += 1
                total_loss_val += accumulated_loss.item()

        avg_loss = total_loss_val / seqs_processed if seqs_processed > 0 else 0.0
        print(f"[Pretrain Epoch {epoch+1}] sequences={seqs_processed} avg_loss={avg_loss:.6f}")

def pretrain_policy_setwise(policy, optimizer, pretraining_samples, env, epochs=25, device='cpu'):
    """
    Refines pre-training using Set-wise MLE. At each step, the model is trained
    on its own greedy prediction, reinforcing its confidence.
    """
    print("Starting Policy Pre-training (Phase 2: Set-wise MLE)...")
    policy.to(device)
    policy.train()
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"--- Set-wise Pretrain Epoch {epoch+1}/{epochs} ---")
        random.shuffle(pretraining_samples)
        seqs_processed = 0
        total_loss_val = 0.0

        for graph, seed_node, bfs_sequence in tqdm(pretraining_samples, desc=f"Set-wise Epoch {epoch+1}"):
            # The target is the full set of nodes, not the sequence
            target_subgraph_S = set(bfs_sequence)
            
            # Initialize environment
            env.current_graph = graph.to(device)
            env.initial_node = seed_node
            env._precompute_graph_structures()

            current_subgraph_nodes = {seed_node}
            accumulated_loss = 0
            
            # We iterate until we've built a subgraph of the target size
            for t in range(len(target_subgraph_S) - 1):
                obs = env._make_obs(current_subgraph_nodes)
                boundary_global = [obs['local_to_global'][i] for i in obs['candidate_nodes']]
                if not boundary_global:
                    break

                S_indices = [obs['local_to_global'].index(n) for n in sorted(list(current_subgraph_nodes))]

                # Get the model's logits (raw scores)
                _, all_scores, _ = policy(obs['x_aug'], obs['A_norm'], obs['candidate_nodes'], S_indices)
                action_logits = all_scores[:-1]

                if action_logits.numel() == 0:
                    break

                # --- THIS IS THE KEY DIFFERENCE ---
                # 1. The target is the model's own most-confident prediction (greedy choice)
                target_node_index = torch.argmax(action_logits).item()
                target_tensor = torch.tensor([target_node_index], dtype=torch.long, device=device)
                
                # 2. Get the actual node ID for the next step
                target_node_global = boundary_global[target_node_index]
                # --- END OF DIFFERENCE ---

                step_loss = loss_fn(action_logits.unsqueeze(0), target_tensor)
                accumulated_loss += step_loss

                # Teacher-force using the model's own greedy choice
                current_subgraph_nodes.add(target_node_global)
                
                # Stop if we've already generated all nodes in the target set
                if current_subgraph_nodes == target_subgraph_S:
                    break

            # Backpropagate after the full sequence
            if isinstance(accumulated_loss, torch.Tensor) and accumulated_loss.item() > 0:
                optimizer.zero_grad()
                accumulated_loss.backward()
                optimizer.step()
                seqs_processed += 1
                total_loss_val += accumulated_loss.item()

        avg_loss = total_loss_val / seqs_processed if seqs_processed > 0 else 0.0
        print(f"[Set-wise Epoch {epoch+1}] sequences={seqs_processed} avg_loss={avg_loss:.6f}")


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
                            log_every = 10,
                            model_path = "./models/policy_model.pt"):
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

        while not done:
            x_aug = obs['x_aug']
            A_norm = obs['A_norm']
            candidates = list(obs['candidate_nodes'])
            S_indices = [obs['local_to_global'].index(n) for n in env.S]

            probs, _, value = policy(x_aug, A_norm, candidates, S_indices)
            
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
                q_est, rollout_rewards = env.estimate_Q(S_after_action, policy, M=rollout_M, max_rollout_steps=rollout_max_steps)
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
        loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
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

    print("Training complete, saving model.")

    # Create directory if it does not exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save
    torch.save(policy.state_dict(), model_path)

