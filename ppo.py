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

        # Apply GAT layers with attention
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))

        # Global mean pooling to obtain a fixed-size graph representation
        graph_embedding = global_mean_pool(h, batch)
        graph_embedding = graph_embedding.view(-1)

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

        # Unified policy network
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
        return self.policy_net(features), self.value_net(features)

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


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GNNActorCriticNetwork(self.action_space)

def train_interpretable_gnn(
    gnn_model,
    dataloader,
    batch_size: int = 32,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    device: str = 'cuda'
):
    
    env = GNNInterpretEnvironment(gnn_model, dataloader, batch_size, device)

    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=batch_size * 4,  # Number of steps per update
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
        log_interval=10
    )

    obs, _ = env.reset()
    action, _states = model.predict(obs)

    print(action.shape)

    return model
