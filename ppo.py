import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, List, Optional, Tuple, Type, Union

class GNNPolicy(ActorCriticPolicy):
    """Custom policy network that can process graph data"""
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # Get feature dimensions from observation space
        self.input_dim = observation_space['x'].shape[1]
        self.hidden_dim = 64
        
        # GNN layers
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        
        # MLP for node scores
        self.node_policy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # MLP for edge scores
        self.edge_policy = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Value function
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward_gnn(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process the graph data through GNN layers"""
        x = obs['x'].to(self.device)
        edge_index = obs['edge_index'].to(self.device)
        batch = obs['batch'].to(self.device)
        
        # GNN layers
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        return h, batch

    def forward(self, 
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy
        Returns: actions, values, log_probs
        """
        # Convert numpy arrays to tensors
        obs_tensors = {
            'x': torch.FloatTensor(obs['x']).to(self.device),
            'edge_index': torch.LongTensor(obs['edge_index']).to(self.device),
            'batch': torch.LongTensor(obs['batch']).to(self.device)
        }
        
        # Get node embeddings
        h, batch = self.forward_gnn(obs_tensors)
        
        # Get node scores
        node_scores = self.node_policy(h).squeeze(-1)
        
        # Get edge scores
        row, col = obs_tensors['edge_index']
        edge_features = torch.cat([h[row], h[col]], dim=1)
        edge_scores = self.edge_policy(edge_features).squeeze(-1)
        
        # Combine node and edge scores
        actions_raw = torch.cat([node_scores, edge_scores])
        
        # Use Beta distribution for valid probability range (0,1)
        alpha = F.softplus(actions_raw) + 1
        beta = F.softplus(-actions_raw) + 1

        dist = Beta(alpha, beta)
        actions = dist.mean if deterministic else dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute value function
        graph_embedding = global_mean_pool(h, batch)
        values = self.value_net(graph_embedding).squeeze(-1)

        return actions.detach().cpu().numpy(), values.detach().cpu().numpy(), log_probs.detach().cpu().numpy()


    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute value function prediction"""
        obs_tensors = {
            'x': torch.FloatTensor(obs['x']).to(self.device),
            'edge_index': torch.LongTensor(obs['edge_index']).to(self.device),
            'batch': torch.LongTensor(obs['batch']).to(self.device)
        }
        
        h, batch = self.forward_gnn(obs_tensors)
        graph_embedding = global_mean_pool(h, batch)
        values = self.value_net(graph_embedding).squeeze(-1)
        
        return values

def train_interpretable_gnn(
    gnn_model,
    dataloader,
    batch_size: int = 32,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    device: str = 'cuda'
):
    
    env = DummyVecEnv([lambda: GNNInterpretEnvironment(gnn_model, dataloader, batch_size, device)])

    # Initialize PPO with custom policy
    model = PPO(
        policy=GNNPolicy,
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
    
    return model
