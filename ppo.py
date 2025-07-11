import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env import GNNInterpretEnvironment
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom GNN feature extractor using Graph Attention Networks (GAT).
    """
    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 64, heads: int = 4):
        super(GNNFeatureExtractor, self).__init__(observation_space, hidden_dim)

        self.hidden_dim = hidden_dim
        input_dim = observation_space.spaces["x"].shape[1]

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)  # Attention layer 1
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)  # Attention layer 2

        self.layer_norm1 = nn.LayerNorm(hidden_dim * heads)  # Normalization after GAT1
        self.layer_norm2 = nn.LayerNorm(hidden_dim)  # Normalization after GAT
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * int(observation_space.spaces["batch"].high_repr), hidden_dim),
            nn.ReLU()
        )

        print(f"MLP Size: {hidden_dim * int(observation_space.spaces['batch'].high_repr)}")
        print("Observation space batch size: ", observation_space.spaces["batch"].high_repr)

    def forward(self, observations: dict) -> torch.Tensor:

        """
        Extract meaningful graph embeddings.
        """
        x = observations["x"].to(torch.float32).squeeze()  # Node features
        edge_index = observations["edge_index"].to(torch.int64).squeeze()  # Edge indices
        batch = observations["batch"].to(torch.int64).squeeze()  # Batch mapping

          # Check for NaNs or Infs in inputs
        assert not torch.isnan(x).any(), "NaN detected in node features (x)!"
        assert not torch.isinf(x).any(), "Inf detected in node features (x)!"
        assert (edge_index >= 0).all(), "Negative indices detected in edge_index!"
        assert (edge_index < x.shape[0]).all(), "Invalid indices in edge_index!"

        # Apply GAT layers with attention
        h = self.gat1(x, edge_index)
        print("Output after GAT1 before activation:", h)
        assert not torch.isnan(h).any(), "NaN detected after GAT1!"

        h = self.layer_norm1(h)  # Normalize after GAT1
        h = F.elu(h)  # Apply activation
        assert not torch.isnan(h).any(), "NaN detected after ELU activation!"

        h = self.gat2(h, edge_index)
        h = self.layer_norm2(h)  # Normalize after GAT2
        h = F.leaky_relu(h, negative_slope=0.01)

        # Global mean pooling to obtain a fixed-size graph representation
        graph_embedding = global_mean_pool(h, batch)
        graph_embedding = graph_embedding.view(-1)

        assert not torch.isnan(graph_embedding).any(), "NaN detected in graph_embedding!"

        return self.mlp(graph_embedding)  # Return processed features

class GNNActorCriticNetwork(nn.Module):
    """
    Custom Actor-Critic Network using GAT-based feature extractor.
    """
    def __init__(self, action_space):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = action_space.shape[0]
        self.latent_dim_vf = 1

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim_pi),  # Output action size
            nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs={"hidden_dim": 64, "heads": 4}
        )
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for each observation in the batch individually.
        
        Args:
            obs: Dictionary of batched observations (batch_size=128)
            actions: Batched actions (batch_size=128)
        """
        batch_size = obs["x"].shape[0]
        
        # Lists to store individual results
        all_values = []
        all_log_probs = []
        all_entropies = []

        # Process each observation individually
        for i in range(batch_size):
            # Create a single-batch observation
            single_obs = {
                "x": obs["x"][i:i+1],  # Keep the batch dimension but size=1
                "edge_index": obs["edge_index"][i:i+1],
                "batch": obs["batch"][i:i+1]
            }

            # Extract features
            features = self.extract_features(single_obs)
            
            # Get latent policy and value
            latent_pi, latent_vf = self.mlp_extractor(features)
            
            # Get distribution for single action
            distribution = self._get_action_dist_from_latent(latent_pi)
            
            # Get log probability of single action
            single_action = actions[i:i+1]
            log_prob = distribution.log_prob(single_action)
            
            # Get entropy
            entropy = distribution.entropy()
            
            # Get value
            value = self.value_net(latent_vf)

            # Expand dimensions if necessary
            if entropy.dim() == 0:  # If scalar (0-dim tensor)
                entropy = entropy.unsqueeze(0)  # Make it [1]
            
            # Ensure proper dimensions for values and log_probs too
            if value.dim() == 0:
                value = value.unsqueeze(0)
            if log_prob.dim() == 0:
                log_prob = log_prob.unsqueeze(0)
            
            # Store results
            all_values.append(value)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)

        # Concatenate all results
        values = torch.cat(all_values, dim=0)
        log_probs = torch.cat(all_log_probs, dim=0)
        entropies = torch.cat(all_entropies, dim=0)

        return values, log_probs, entropies

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNActorCriticNetwork(self.action_space)

def train_interpretable_gnn(
    gnn_model,
    dataloader,
    batch_size: int = 32,
    total_timesteps: int = 10000,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    
    env = GNNInterpretEnvironment(gnn_model, dataloader, batch_size, device)

    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=10,  # Number of steps per update
        batch_size=batch_size,   # PPO batch size
        n_epochs=10,             # Number of epochs when optimizing the surrogate loss
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,         # Clipping parameter for PPO
        clip_range_vf=None,     # Clipping parameter for the value function
        ent_coef=0.01,          # Entropy coefficient for exploration
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Maximum norm for gradient clipping
        use_sde=False,          # Whether to use generalized State Dependent Exploration
        sde_sample_freq=-1,     # Sample a new noise matrix every n steps
        verbose=1,
        device=device
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1
    )

    obs, _ = env.reset()
    action, _states = model.predict(obs)

    print(action.shape)

    return model
