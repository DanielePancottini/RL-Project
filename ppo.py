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
from typing import Callable, Dict, List, Optional, Tuple, Type
from gymnasium import spaces

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom GNN feature extractor using Graph Attention Networks (GAT).
    """
    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 64, heads: int = 4):
        super(GNNFeatureExtractor, self).__init__(observation_space, hidden_dim)

        self.hidden_dim = hidden_dim
        self.heads = heads
        input_dim = observation_space.spaces["x"].shape[1]

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)  # Attention layer 1
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)  # Attention layer 2

        self.layer_norm1 = nn.LayerNorm(hidden_dim * heads)  # Normalization after GAT1
        self.layer_norm2 = nn.LayerNorm(hidden_dim)  # Normalization after GAT
        
        print(f"GNNFeatureExtractor will output features of dimension: {self._features_dim}")

    def forward(self, observations: dict) -> torch.Tensor:

        print(f"FeatureExtractor DEBUG: observations['num_nodes'] received shape: {observations['num_nodes'].shape}")

        """
        Extract meaningful graph embeddings.
        """
        # Reconstruct PyG batch
        batch_size_sb3 = observations["x"].shape[0]
        list_of_data = []

        for i in range(batch_size_sb3):
            print(f"FeatureExtractor DEBUG: observations['num_nodes'][{i}] shape before .item(): {observations['num_nodes'][i].shape}")
            num_nodes = int(observations["num_nodes"][i].item())
            num_edges = int(observations["num_edges"][i].item())
            
            x_i = observations["x"][i, :num_nodes, :].contiguous()
            edge_index_i = observations["edge_index"][i, :, :num_edges].contiguous()
            edge_index_i = edge_index_i.to(torch.long)

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
        self.latent_dim_pi = action_space.n
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
            # self.log_std = None
            # For discrete action spaces, this parameter is not needed,
            # but SB3's logger currently expects a Tensor.
            # Setting a dummy parameter to avoid the TypeError.
            self.log_std = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device))

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

        distribution = torch.distributions.Categorical(logits=latent_pi)
        if deterministic:
            actions = torch.argmax(distribution.logits, dim=-1)
        else:
            actions = distribution.sample()
        
        # Calculate log probability of the sampled actions
        log_prob = distribution.log_prob(actions)

        # Get value estimates
        values = self.mlp_extractor.forward_critic(features).squeeze(1) # Ensure correct shape

        return actions, values, log_prob

def train_interpretable_gnn(baseline_gnn, env_dataloader, n_envs, max_nodes, max_edges, node_feature_dim, device, max_episode_steps):
    
    # Create the vectorized environment
    env = DummyVecEnv([
        lambda: GNNInterpretEnvironment(
            gnn_model=baseline_gnn,
            single_graph_dataloader=env_dataloader,
            max_nodes=max_nodes,
            max_edges=max_edges,
            node_feature_dim=node_feature_dim,
            device=device,
            max_steps_per_episode=max_episode_steps # Pass to environment
        )
        for _ in range(n_envs)
    ])

    
    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNActorCriticPolicy, # ⭐ IMPORTANT: Use your custom policy here! ⭐
        env=env,
        learning_rate=3e-4, # Use the passed learning_rate
        n_steps=64,               # Number of steps collected per environment before update
        batch_size=256,             # Minibatch size for SGD updates during policy training
        n_epochs=10,                # Number of epochs for PPO updates per rollout
        gamma=0.99,                 # Discount factor
        gae_lambda=0.95,            # GAE parameter
        clip_range=0.2,             # PPO clipping range
        ent_coef=0.01,              # Entropy coefficient for exploration
        vf_coef=0.5,                # Value function coefficient
        max_grad_norm=0.5,          # Max gradient norm for clipping
        tensorboard_log="./ppo_tensorboard/",
        verbose=1,                  # Verbosity level (1 for training progress)
        device=device,              # Device for computations ('cpu' or 'cuda')
        policy_kwargs=dict(         # Pass arguments to your custom policy's __init__
            features_extractor_class=GNNFeatureExtractor, # Specify your GNN feature extractor
            features_extractor_kwargs=dict(hidden_dim=64, heads=4) # Pass args to GNNFeatureExtractor
        )
    )
    
    print(f"PPO Model created. Training for {model._total_timesteps} timesteps...")
    # total_timesteps refers to the total number of environment steps (sum across all environments)
    model.learn(total_timesteps=50000) # Example: Train for 500k environment steps

    return model
