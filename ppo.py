import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from env import GNNInterpretEnvironment
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
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
        
        print(f"GNNFeatureExtractor will output features of dimension: {self._features_dim}")

    def forward(self, observations: dict) -> torch.Tensor:

        """
        Extract meaningful graph embeddings.
        """
        # Reconstruct PyG batch
        batch_size_sb3 = observations["x"].shape[0]
        list_of_data = []

        for i in range(batch_size_sb3):
            num_nodes = observations["num_nodes"][i].item()
            num_edges = observations["num_edges"][i].item()
            
            x_i = observations["x"][i, :num_nodes, :].contiguous()
            edge_index_i = observations["edge_index"][i, :, :num_edges].contiguous()

            # Handle case where a graph has no edges
            if edge_index_i.numel() == 0:
                edge_index_i = torch.empty((2, 0), dtype=torch.long, device=x_i.device)

            list_of_data.append(Data(x=x_i, edge_index=edge_index_i))

        # Create a single PyTorch Geometric Batch from the list of Data objects
        pyg_batch = Batch.from_data_list(list_of_data)

        # Extract components for GNN layers
        x, edge_index, batch_map = pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch

        # ❗ CHANGE 4: Move input assertions to after PyG batch construction ❗
        assert not torch.isnan(x).any(), "NaN detected in node features (x)!"
        assert not torch.isinf(x).any(), "Inf detected in node features (x)!"
        # Check edge_index values against the node count in the _combined_ batch
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
        graph_embeddings = global_mean_pool(h, batch_map) # [batch_size_sb3, self.hidden_dim]


        assert not torch.isnan(graph_embeddings).any(), "NaN detected in graph_embedding!"

        return graph_embeddings  # Return features

class GNNActorCriticNetwork(nn.Module):
    """
    Custom Actor-Critic Network using GAT-based feature extractor.
    """
    def __init__(self, action_space, features_dim: int = 64):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = action_space.shape[0]
        self.latent_dim_vf = 1

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim_pi)  # Output action size
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.latent_dim_vf)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        features_extractor_class: Type[BaseFeaturesExtractor] = GNNFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, any]] = None,
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
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=None,
            *args,
            **kwargs,
        )

        if isinstance(action_space, spaces.Box): # Only for continuous action spaces
            self.log_std = nn.Parameter(torch.zeros(self.action_space.shape[0], dtype=torch.float32))
        else:
            # For discrete action spaces, this parameter is not needed
            self.log_std = None

    def _build_mlp_extractor(self) -> None:
        # Get the feature dimension from the already initialized features_extractor
        features_dim = self.features_extractor.features_dim # This refers to self._features_dim from BaseFeaturesExtractor

        # Instantiate your custom ActorCriticNetwork
        self.mlp_extractor = GNNActorCriticNetwork(self.action_space, features_dim)


    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get latent features (graph embeddings) from the feature extractor
        features = self.features_extractor(obs)
        
        # Get latent policy and value (output of your GNNActorCriticNetwork's MLPs)
        latent_pi, latent_vf = self.mlp_extractor(features) # Call the custom network

        # Create the action distribution
        # _get_action_dist_from_latent is a method of the base ActorCriticPolicy
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Calculate log_prob and entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Get the value estimates (output from value_net)
        values = self.mlp_extractor.forward_critic(features).squeeze(1) # Ensure values is [batch_size]
        
        return values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> DiagGaussianDistribution:
        mean_actions = latent_pi
        
        # For continuous actions, use the learned log_std
        # Clamp for numerical stability: log_std usually clamped between -20 and 2
        log_std = self.log_std.expand_as(mean_actions).clamp(min=-20.0, max=2.0)
        std_actions = torch.exp(log_std)
        
        return DiagGaussianDistribution(mean_actions, std_actions)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in the policy.

        :param obs: Observation dictionary from the environment.
        :param deterministic: Whether to sample or use deterministic actions.
        :return: action, value_estimates, log_prob
        """
        # Features from the GNNFeatureExtractor
        features = self.features_extractor(obs)

        # Latent policy and value from your custom actor-critic network
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Create the action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Sample action (or use deterministic)
        actions = distribution.get_actions(deterministic=deterministic)
        
        # Calculate log probability of the sampled actions
        log_prob = distribution.log_prob(actions)

        # Get value estimates
        values = self.mlp_extractor.forward_critic(features).squeeze(1) # Ensure correct shape

        return actions, values, log_prob

def train_interpretable_gnn(
    gnn_model,
    dataloader,
    batch_size: int = 32,
    total_timesteps: int = 10000,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    # Create a function that returns an environment instance
    def make_env():
        return GNNInterpretEnvironment(gnn_model, dataloader, batch_size, device)

    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: GNNInterpretEnvironment(gnn_model, dataloader, 1, device)] * batch_size) 

    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,  # Number of steps per update
        batch_size=64,   # PPO batch size
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
        device=device,
        policy_kwargs=dict(
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(hidden_dim=64, heads=4)
        )
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1
    )

    obs, _ = env.reset()
    action, _states = model.predict(obs)

    print(action.shape) # This will be (n_envs, action_space_dim)

    return model
