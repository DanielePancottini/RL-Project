import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy 
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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
        
        #input_dim = node features + node mask + is starting node mask
        node_features_dim = observation_space.spaces["x"].shape[1]
        input_dim = node_features_dim + 1 + 1

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
        with torch.no_grad():
            batch_size_sb3 = observations["x"].shape[0]
            list_of_data = []

            for i in range(batch_size_sb3):
                num_nodes = int(observations["num_nodes"][i].item())
                num_edges = int(observations["num_edges"][i].item())
                
                x_i = observations["x"][i, :num_nodes, :].contiguous()

                # Extract and unpad node_mask and is_start_node_mask, then unsqueeze to [num_nodes, 1]
                node_mask_i = observations["node_mask"][i, :num_nodes].unsqueeze(1).contiguous()
                is_start_node_mask_i = observations["is_start_node_mask"][i, :num_nodes].unsqueeze(1).contiguous()

                # Concatenate all node-level features for the GNN input
                x_augmented_i = torch.cat([x_i, node_mask_i, is_start_node_mask_i], dim=-1)

                # Extract and unpad edge_index
                edge_index_i = observations["edge_index"][i, :, :num_edges].contiguous()
                edge_index_i = edge_index_i.to(torch.long)

                # Handle case where a graph has no edges
                if edge_index_i.numel() == 0:
                    edge_index_i = torch.empty((2, 0), dtype=torch.long, device=x_i.device)

                list_of_data.append(Data(x=x_augmented_i, edge_index=edge_index_i))

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
            nn.Linear(features_dim, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim_pi)
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim_vf)
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
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # For discrete action spaces, use Categorical distribution
        distribution = torch.distributions.Categorical(logits=latent_pi)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        values = self.mlp_extractor.forward_critic(features).squeeze(1)
        
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
            node_features_dim=node_feature_dim,
            device=device,
            max_steps_per_episode=max_episode_steps # Pass to environment
        )
        for _ in range(n_envs)
    ])

    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNActorCriticPolicy, # ⭐ IMPORTANT: Use your custom policy here! ⭐
        env=env,
        learning_rate=3e-5, # Use the passed learning_rate
        n_steps=2048,               # Number of steps collected per environment before update
        batch_size=512,             # Minibatch size for SGD updates during policy training
        n_epochs=8,                # Number of epochs for PPO updates per rollout
        gamma=0.95,                 # Discount factor
        gae_lambda=0.92,            # GAE parameter
        clip_range=0.15,             # PPO clipping range
        target_kl=0.03,  # Add KL early stopping
        ent_coef=0.01,              # Entropy coefficient for exploration
        vf_coef=0.7,                # Value function coefficient
        max_grad_norm=0.3,          # Max gradient norm for clipping
        tensorboard_log="./ppo_tensorboard/",
        verbose=1,                  # Verbosity level (1 for training progress)
        device=device,              # Device for computations ('cpu' or 'cuda')
        policy_kwargs=dict(         # Pass arguments to your custom policy's __init__
            features_extractor_class=GNNFeatureExtractor, # Specify your GNN feature extractor
            features_extractor_kwargs=dict(hidden_dim=512, heads=4) # Pass args to GNNFeatureExtractor
        )
    )
    
    print(f"PPO Model created. Training for {model._total_timesteps} timesteps...")
    # total_timesteps refers to the total number of environment steps (sum across all environments)
    model.learn(total_timesteps=4096) # Example: Train for 500k environment steps

    return model
