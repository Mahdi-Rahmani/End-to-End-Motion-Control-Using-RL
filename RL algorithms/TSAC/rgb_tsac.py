#!/usr/bin/env python

# Improved T-SAC implementation following the paper more closely
# This code uses the RGB birdeye view as state input with a Transformer-based critic
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gym
import gym_carla
import carla
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import collections
import random
import time
from datetime import datetime
import cv2
import traceback
import sys
import matplotlib.pyplot as plt

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str, help='CARLA server host')
    parser.add_argument('--port', default=3000, type=int, help='CARLA server port')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic manager port')
    parser.add_argument('--sync', action='store_true', help='Synchronous mode')
    parser.add_argument('--no-rendering', action='store_true', help='No rendering mode')
    parser.add_argument('--episodes', default=100, type=int, help='Number of episodes to train')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to load model checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', default='./logs', type=str, help='Directory for tensorboard logs')
    parser.add_argument('--recovery-timeout', default=5, type=int, help='Time to wait after an error before retrying')
    parser.add_argument('--pedestrians', default=5, type=int, help='Number of pedestrians for jaywalking')
    
    # T-SAC specific parameters - tuned to match paper
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--policy-lr', default=3e-4, type=float, help='Policy learning rate')
    parser.add_argument('--critic-lr', default=1e-4, type=float, help='Critic learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--tau', default=0.005, type=float, help='Soft update coefficient')
    parser.add_argument('--alpha', default=0.2, type=float, help='Temperature parameter for entropy')
    parser.add_argument('--auto-alpha', action='store_true', help='Automatically tune alpha')
    parser.add_argument('--buffer-size', default=50000, type=int, help='Replay buffer size')
    parser.add_argument('--min-length', default=1, type=int, help='Minimum sequence length')
    parser.add_argument('--max-length', default=16, type=int, help='Maximum sequence length')
    parser.add_argument('--hidden-dim', default=128, type=int, help='Hidden dimension size')
    parser.add_argument('--utd-ratio', default=0.25, type=float, help='Update to Data ratio')
    parser.add_argument('--critic-updates', default=100, type=int, help='Critic updates per policy update batch')
    parser.add_argument('--policy-updates', default=20, type=int, help='Policy updates per batch')
    
    return parser.parse_args()

# CNN feature extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        self.input_shape = input_shape  # (3, 84, 84) for RGB
        
        # CNN layers - simplified
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the CNN output
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        self.feature_size = convw * convh * 64
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(-1, self.feature_size)

# Policy Network with enhanced design from the paper
class TSACPolicyNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super(TSACPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extractor
        self.features = CNNFeatureExtractor(input_shape)
        feature_size = self.features.feature_size
        
        # Mean network with layer normalization
        self.mean_fc1 = nn.Linear(feature_size, hidden_dim)
        self.mean_ln1 = nn.LayerNorm(hidden_dim)
        self.mean_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_ln2 = nn.LayerNorm(hidden_dim)
        self.mean_out = nn.Linear(hidden_dim, action_dim)
        
        # Variance network with layer normalization and mean input
        self.var_fc1 = nn.Linear(feature_size + action_dim, hidden_dim)  # +action_dim for mean input
        self.var_ln1 = nn.LayerNorm(hidden_dim)
        self.var_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.var_ln2 = nn.LayerNorm(hidden_dim)
        self.log_std_out = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with small values as per the paper
        nn.init.xavier_uniform_(self.mean_out.weight, gain=0.01)
        nn.init.constant_(self.mean_out.bias, 0)
        
    def forward(self, x):
        features = self.features(x)
        
        # Mean network
        mean_x = F.leaky_relu(self.mean_ln1(self.mean_fc1(features)))
        mean_x = F.leaky_relu(self.mean_ln2(self.mean_fc2(mean_x)))
        mean = self.mean_out(mean_x)
        
        # Variance network (with mean input) - key detail from the paper
        # Detach mean to prevent backprop through variance affecting mean
        var_input = torch.cat([features, mean.detach()], dim=1)
        var_x = F.leaky_relu(self.var_ln1(self.var_fc1(var_input)))
        var_x = F.leaky_relu(self.var_ln2(self.var_fc2(var_x)))
        log_std = self.log_std_out(var_x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Sample action from normal distribution
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, accounting for the transformation
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for tanh
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x, start_idx=0):
        seq_len = x.size(1)
        return x + self.pe[start_idx:start_idx+seq_len].unsqueeze(0)

# Simplified Transformer-based Critic Network following the paper
class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_seq_len=16):
        super(TransformerCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # State and action embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Simplified transformer with 2 heads, 1 layer - as per our discussion
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,  # Reduced to 2 heads
            dim_feedforward=hidden_dim*2,  # Simplified
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=1  # Reduced to 1 layer for simplicity
        )
        
        # Output heads for different sequence lengths - key paper concept
        # Each head predicts Q(s_t, a_t, ..., a_t+i)
        self.q_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(max_seq_len)])
    
    def forward(self, state, actions):
        """
        Args:
            state: (batch_size, state_dim) tensor - initial state
            actions: (batch_size, seq_len, action_dim) tensor - sequence of actions
        
        Returns:
            q_values: list of (batch_size, 1) tensors for each subsequence length
        """
        batch_size, seq_len, _ = actions.shape
        
        # Embed state and expand to match sequence length
        state_embed = self.state_embedding(state).unsqueeze(1)
        state_embed = state_embed.expand(-1, seq_len, -1)
        
        # Embed actions
        action_embed = self.action_embedding(actions)
        
        # Combine state and action embeddings
        combined = state_embed + action_embed
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Causal mask (lower triangular) to ensure predictions only depend on past actions
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(actions.device)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(combined, mask=mask)
        
        # Get Q-values for each subsequence length
        # This is the key insight from the paper - predict values for each subsequence
        q_values = []
        for i in range(min(seq_len, len(self.q_outputs))):
            q_i = self.q_outputs[i](transformer_output[:, i])
            q_values.append(q_i)
        
        return q_values

# Trajectory Replay Buffer
class TrajectoryReplayBuffer:
    def __init__(self, capacity, max_episode_length=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.max_episode_length = max_episode_length
        
    def add_trajectory(self, trajectory):
        """Add a complete trajectory to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
        else:
            self.buffer[self.position] = trajectory
            self.position = (self.position + 1) % self.capacity
    
    def sample_episodes(self, batch_size, min_length, max_length):
        """Sample random episodes and starting points."""
        # Filter trajectories that are long enough
        valid_trajectories = [traj for traj in self.buffer if len(traj) >= min_length]
        
        if len(valid_trajectories) == 0:
            return None
        
        # Sample batch_size episodes
        sampled_episodes = random.sample(valid_trajectories, min(batch_size, len(valid_trajectories)))
        
        # Process each episode
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_sequence_length = []
        
        for episode in sampled_episodes:
            # Choose a random starting point and sequence length
            episode_length = len(episode)
            max_start_idx = episode_length - min_length
            
            if max_start_idx < 0:
                continue
                
            start_idx = random.randint(0, max_start_idx)
            seq_length = min(max_length, episode_length - start_idx)
            
            # Extract sequence
            sequence = episode[start_idx:start_idx + seq_length]
            states, actions, rewards, next_states, dones = zip(*sequence)
            
            batch_states.append(states[0])  # Initial state
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
            batch_sequence_length.append(seq_length)
        
        # Convert to tensors
        states = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
        
        # Process sequences
        max_seq = max(batch_sequence_length)
        
        # Pad sequences
        padded_actions = []
        padded_rewards = []
        padded_next_states = []
        padded_dones = []
        
        for i in range(len(batch_sequence_length)):
            # Pad actions
            actions_seq = list(batch_actions[i])
            actions_seq.extend([np.zeros_like(actions_seq[0])] * (max_seq - len(actions_seq)))
            padded_actions.append(actions_seq)
            
            # Pad rewards
            rewards_seq = list(batch_rewards[i])
            rewards_seq.extend([0.0] * (max_seq - len(rewards_seq)))
            padded_rewards.append(rewards_seq)
            
            # Pad next_states
            next_states_seq = list(batch_next_states[i])
            next_states_seq.extend([np.zeros_like(next_states_seq[0])] * (max_seq - len(next_states_seq)))
            padded_next_states.append(next_states_seq)
            
            # Pad dones
            dones_seq = list(batch_dones[i])
            dones_seq.extend([True] * (max_seq - len(dones_seq)))
            padded_dones.append(dones_seq)
        
        actions = torch.tensor(np.array(padded_actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(padded_rewards), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(padded_next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(padded_dones), dtype=torch.float32, device=device)
        lengths = torch.tensor(batch_sequence_length, dtype=torch.long, device=device)
        
        return states, actions, rewards, next_states, dones, lengths
    
    def __len__(self):
        return len(self.buffer)

# T-SAC Agent
class TSACAgent:
    def __init__(self, state_shape, action_dim, args):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.auto_alpha = args.auto_alpha
        self.batch_size = args.batch_size
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.hidden_dim = args.hidden_dim
        self.critic_updates_per_batch = args.critic_updates
        self.policy_updates_per_batch = args.policy_updates
        self.utd_ratio = args.utd_ratio
        
        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(state_shape).to(device)
        feature_size = self.feature_extractor.feature_size
        
        # Initialize policy network
        self.policy_net = TSACPolicyNetwork(
            state_shape, 
            action_dim,
            hidden_dim=self.hidden_dim
        ).to(device)
        
        # Initialize twin transformer critics
        self.critic1 = TransformerCritic(
            state_dim=feature_size, 
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            max_seq_len=args.max_length
        ).to(device)
        
        self.critic2 = TransformerCritic(
            state_dim=feature_size, 
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            max_seq_len=args.max_length
        ).to(device)
        
        # Initialize target critics
        self.target_critic1 = TransformerCritic(
            state_dim=feature_size, 
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            max_seq_len=args.max_length
        ).to(device)
        
        self.target_critic2 = TransformerCritic(
            state_dim=feature_size, 
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            max_seq_len=args.max_length
        ).to(device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Set target networks to evaluation mode
        self.target_critic1.eval()
        self.target_critic2.eval()
        
        # Initialize optimizers with different learning rates as per paper
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.policy_lr)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=args.critic_lr)
        
        # If auto_alpha, initialize log_alpha and optimizer
        if self.auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.policy_lr)
        
        # Initialize trajectory replay buffer
        self.memory = TrajectoryReplayBuffer(args.buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
        self.updates_done = 0
        
        # Current episode buffer
        self.current_episode = []
        
        # For scaling our continuous actions
        self.action_scaling = torch.tensor([2.0, 0.6], device=device)  # [acc_range, steer_range]
        self.action_bias = torch.tensor([1.5, 0.0], device=device)
    
    def select_action(self, state, evaluate=False):
        """Select action from the policy."""
        state = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy_net(state)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.policy_net.sample(state)
        
        scaled_action = action * self.action_scaling + self.action_bias
        
        return scaled_action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in current episode buffer."""
        self.current_episode.append((state, action, reward, next_state, done))
        
        # Store sub-trajectories during episode
        if len(self.current_episode) >= self.min_length + 5:
            if len(self.current_episode) % 10 == 0:
                sub_traj = self.current_episode[-self.min_length-5:]
                self.memory.add_trajectory(sub_traj)
        
        if done:
            if len(self.current_episode) >= self.min_length:
                self.memory.add_trajectory(self.current_episode)
                print(f"Added trajectory of length {len(self.current_episode)}. Total: {len(self.memory)}")
            else:
                print(f"Trajectory too short ({len(self.current_episode)} < {self.min_length}). Not storing.")
            self.current_episode = []
    
    def compute_n_step_returns(self, rewards, next_values, dones, seq_lengths):
        """
        Compute N-step returns for all subsequence lengths as per paper.
        
        Returns:
            returns: Dictionary mapping step number to (batch_size,) tensor of returns
        """
        batch_size, max_seq_len = rewards.size()
        n_step_returns = {}
        
        # For each subsequence length
        for n in range(1, max_seq_len + 1):
            if n > max_seq_len:
                continue
                
            n_returns = torch.zeros(batch_size, device=rewards.device)
            
            for b in range(batch_size):
                if n > seq_lengths[b]:
                    continue
                    
                # Calculate n-step return
                ret = 0
                for i in range(min(n, seq_lengths[b])):
                    ret += (self.gamma ** i) * rewards[b, i]
                    if dones[b, i]:
                        break
                
                # Add bootstrapped value
                if n < seq_lengths[b] and not dones[b, n-1]:
                    ret += (self.gamma ** n) * next_values[b]
                
                n_returns[b] = ret
            
            n_step_returns[n] = n_returns
            
        return n_step_returns
        
    def update_critic(self, batch):
        """Update critic networks with gradient-level averaging for N-step returns."""
        states, actions, rewards, next_states, dones, seq_lengths = batch
        batch_size, max_seq_len = rewards.size()
        
        # Extract state features
        with torch.no_grad():
            state_features = self.feature_extractor(states)
        
            # Get final states for bootstrapping
            final_indices = seq_lengths - 1
            final_states = torch.stack([
                next_states[b, idx] for b, idx in enumerate(final_indices)
            ])
            
            final_state_features = self.feature_extractor(final_states)
            
            # Get actions for final states
            final_actions, final_log_probs, _ = self.policy_net.sample(final_states)
            final_action_seqs = final_actions.unsqueeze(1)
            
            # Get target critic values
            target_q1_final = self.target_critic1(final_state_features, final_action_seqs)[0].squeeze(-1)
            target_q2_final = self.target_critic2(final_state_features, final_action_seqs)[0].squeeze(-1)
            min_target_q = torch.min(target_q1_final, target_q2_final)
            
            # Subtract entropy term
            final_values = min_target_q - self.alpha * final_log_probs.squeeze(-1)
        
        # Compute N-step returns
        n_step_returns = self.compute_n_step_returns(rewards, final_values, dones, seq_lengths)
        
        # Forward pass through critics
        critic1_values = self.critic1(state_features, actions)
        critic2_values = self.critic2(state_features, actions)
        
        # Compute critic losses for each subsequence length (key paper insight)
        critic1_losses = []
        critic2_losses = []
        
        for n in range(1, min(max_seq_len, len(critic1_values)) + 1):
            if n in n_step_returns:
                target_n = n_step_returns[n]
                
                q1_n = critic1_values[n-1].squeeze(-1)
                q2_n = critic2_values[n-1].squeeze(-1)
                
                valid_mask = seq_lengths >= n
                
                if valid_mask.sum() > 0:
                    critic1_n_loss = F.mse_loss(q1_n[valid_mask], target_n[valid_mask])
                    critic2_n_loss = F.mse_loss(q2_n[valid_mask], target_n[valid_mask])
                    
                    critic1_losses.append(critic1_n_loss)
                    critic2_losses.append(critic2_n_loss)
        
        # Average losses at gradient level as per paper
        if len(critic1_losses) > 0 and len(critic2_losses) > 0:
            critic1_loss = sum(critic1_losses) / len(critic1_losses)
            critic2_loss = sum(critic2_losses) / len(critic2_losses)
            critic_loss = critic1_loss + critic2_loss
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            
            return critic_loss.item()
        
        return 0
    
    def update_policy(self, batch):
        """Update policy network."""
        states, actions, rewards, next_states, dones, seq_lengths = batch
        
        # Extract state features
        with torch.no_grad():
            state_features = self.feature_extractor(states)
        
        # Freeze critics for policy update
        for param in self.critic1.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
        
        # Sample actions from policy
        new_actions, log_probs, _ = self.policy_net.sample(states)
        
        # Single-action sequence for policy update
        new_action_seqs = new_actions.unsqueeze(1)
        
        # Get Q-values
        q1 = self.critic1(state_features, new_action_seqs)[0].squeeze(-1)
        q2 = self.critic2(state_features, new_action_seqs)[0].squeeze(-1)
        min_q = torch.min(q1, q2)
        
        # Compute policy loss (maximizing Q-value - alpha * log_prob)
        policy_loss = (self.alpha * log_probs.squeeze(-1) - min_q).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha if enabled
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.squeeze(-1) + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Unfreeze critics
        for param in self.critic1.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True
        
        return policy_loss.item()
    
    def update_targets(self):
        """Soft update target networks."""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self):
        """Train the agent using the Update-to-Data ratio and update schedule from the paper."""
        if len(self.memory) < self.batch_size:
            return {"critic_loss": 0, "policy_loss": 0, "alpha": self.alpha}
        
        total_critic_loss = 0
        total_policy_loss = 0
        
        # Sample batch
        batch = self.memory.sample_episodes(
            self.batch_size, 
            self.min_length, 
            self.max_length
        )
        
        if batch is None:
            return {"critic_loss": 0, "policy_loss": 0, "alpha": self.alpha}
        
        # 1. Update critics multiple times (100 in paper)
        for _ in range(self.critic_updates_per_batch):
            critic_loss = self.update_critic(batch)
            total_critic_loss += critic_loss
        
        # 2. Update policy multiple times (20 in paper)
        for _ in range(self.policy_updates_per_batch):
            policy_loss = self.update_policy(batch)
            total_policy_loss += policy_loss
        
        # 3. Update target networks
        self.update_targets()
        
        self.updates_done += 1
        
        return {
            "critic_loss": total_critic_loss / self.critic_updates_per_batch,
            "policy_loss": total_policy_loss / self.policy_updates_per_batch,
            "alpha": self.alpha
        }
    
    def save(self, path):
        """Save the model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'alpha': self.alpha
        }, path)
    
    def load(self, path):
        """Load the model."""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha'].to(device)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = checkpoint['alpha']
            
        self.steps_done = checkpoint['steps_done']
        self.updates_done = checkpoint['updates_done']

# Preprocess the birdeye view observation
def preprocess_birdeye(birdeye):
    # Resize to a smaller size
    resized = cv2.resize(birdeye, (84, 84))
    # Normalize pixel values
    normalized = resized / 255.0
    # Transpose to get channels first (PyTorch format)
    transposed = np.transpose(normalized, (2, 0, 1))  # Shape will be (3, 84, 84)
    return transposed

# Train a single episode
def train_single_episode(env, agent, writer, episode, args, reward_history):
    max_timesteps = 1000
    
    # Reset environment
    print("Resetting environment...")
    sys.stdout.flush()
    obs, info = env.reset()
    
    # Process initial observation
    state = preprocess_birdeye(obs['birdeye'])
    
    episode_reward = 0
    episode_critic_loss = 0
    episode_policy_loss = 0
    episode_steps = 0
    update_info = {"critic_loss": 0, "policy_loss": 0, "alpha": agent.alpha}
    
    # Calculate how often to update based on UTD ratio
    # If UTD ratio is 0.25, we update once every 4 steps
    update_frequency = max(1, int(1 / agent.utd_ratio))
    
    # Episode loop
    for t in range(max_timesteps):
        # Select action
        action = agent.select_action(state)
        
        if t % 50 == 0:
            print(f"Step {t}, Action: {action}")
            sys.stdout.flush()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess_birdeye(next_obs['birdeye'])
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        # Move to next state
        state = next_state
        
        # Update according to the UTD ratio
        if len(agent.memory) >= agent.batch_size and t % update_frequency == 0:
            try:
                update_info = agent.train()
                episode_critic_loss += update_info["critic_loss"]
                episode_policy_loss += update_info["policy_loss"]
            except Exception as e:
                print(f"Error during update: {e}")
                traceback.print_exc()
        
        episode_reward += reward
        episode_steps += 1
        agent.steps_done += 1
        
        # Render
        env.render()
        
        # Check if done
        if done:
            if terminated:
                print(f"Episode terminated after {t+1} steps (collision/lane departure)")
            else:
                print(f"Episode truncated after {t+1} steps (time limit)")
            break
    
    # Log stats
    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Loss/Critic_Loss', episode_critic_loss / max(1, episode_steps // update_frequency), episode)
    writer.add_scalar('Loss/Policy_Loss', episode_policy_loss / max(1, episode_steps // update_frequency), episode)
    writer.add_scalar('Parameters/Alpha', update_info["alpha"], episode)
    writer.add_scalar('Steps/Episode', episode_steps, episode)
    
    # Add to reward history
    reward_history.append(episode_reward)
    
    print(f"\nEpisode {episode+1} | Reward: {episode_reward:.2f} | "
          f"Critic Loss: {episode_critic_loss/max(1, episode_steps // update_frequency):.4f} | "
          f"Policy Loss: {episode_policy_loss/max(1, episode_steps // update_frequency):.4f} | "
          f"Alpha: {update_info['alpha']:.4f} | Steps: {episode_steps}")
    sys.stdout.flush()
    
    return episode_reward

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    # Keep track of rewards for plotting
    reward_history = []
    
    # Parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,  # Reduced for training
        'number_of_walkers': 0,  
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln.*',
        'port': args.port,
        'host': args.host,
        'tm_port': args.tm_port,
        'town': 'Town03',
        'task_mode': 'roundabout',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 15.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
        'pixor_size': 64,
        'pixor': False,
        'sync': args.sync,
        'rendering': not args.no_rendering,
        'jaywalking_pedestrians': True
    }
    
    # Create the gym environment
    env = None
    agent = None
    try:
        print("Creating CARLA environment...")
        sys.stdout.flush()
        
        # Main training loop - recreate environment for each episode
        for episode in range(args.episodes):
            try:
                print(f"\n--- Starting episode {episode+1}/{args.episodes} ---")
                sys.stdout.flush()
                
                # Create a fresh environment for each episode
                if env is not None:
                    print("Closing previous environment...")
                    try:
                        env.close()
                    except Exception as e:
                        print(f"Error closing environment: {e}")
                    env = None
                    time.sleep(args.recovery_timeout)  # Wait for CARLA to stabilize
                
                print(f"Creating new environment for episode {episode+1}...")
                sys.stdout.flush()
                env = gym.make('carla-v0', params=params)
                
                # Define state shape and action space
                state_shape = (3, 84, 84)  # RGB channels for PyTorch (channels, height, width)
                action_dim = 2  # [throttle/brake, steering]
                
                # Create agent (or load existing one)
                if episode == 0 or agent is None:
                    print("Creating T-SAC agent...")
                    agent = TSACAgent(state_shape, action_dim, args)
                    if args.checkpoint:
                        print(f"Loading model from {args.checkpoint}")
                        agent.load(args.checkpoint)
                
                # Single episode training
                print(f"Running episode {episode+1}...")
                sys.stdout.flush()
                
                # Run a single episode with better error handling
                try:
                    train_single_episode(env, agent, writer, episode, args, reward_history)
                except RuntimeError as e:
                    print(f"Runtime error: {e}")
                    traceback.print_exc()
                    continue
                
                print(f"Episode {episode+1} completed successfully")
                sys.stdout.flush()
                
                # Save model after each episode
                if episode % 5 == 0 or episode == args.episodes - 1:
                    save_path = os.path.join(args.save_dir, f"model_episode_{episode+1}.pth")
                    print(f"Saving model to {save_path}")
                    agent.save(save_path)
                
            except KeyboardInterrupt:
                print("\nTraining interrupted by user.")
                break
            except Exception as episode_ex:
                print(f"\nError in episode {episode+1}: {episode_ex}")
                traceback.print_exc()
                sys.stdout.flush()
                continue
                
        # Final cleanup
        if env is not None:
            env.close()
        
        # Save final model
        try:
            final_model_path = os.path.join(args.save_dir, "final_model.pth")
            agent.save(final_model_path)
            print(f"Training completed. Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving final model: {e}")
        
        # Plot and save the reward history
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(reward_history) + 1), reward_history)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Reward History')
            plt.grid(True)
            reward_plot_path = os.path.join(args.log_dir, 'reward_history.png')
            plt.savefig(reward_plot_path)
            print(f"Reward history plot saved to {reward_plot_path}")
            
            # Save rewards to CSV
            import pandas as pd
            rewards_df = pd.DataFrame(reward_history, columns=['reward'])
            rewards_df.to_csv(os.path.join(args.log_dir, 'reward_history.csv'))
            
        except Exception as e:
            print(f"Error creating reward plot: {e}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        if env is not None:
            try:
                print("Closing CARLA environment...")
                env.close()
                print("CARLA environment closed")
            except Exception as close_ex:
                print(f"Error closing environment: {close_ex}")
        writer.close()
        print("Clean up completed.")

if __name__ == "__main__":
    main()