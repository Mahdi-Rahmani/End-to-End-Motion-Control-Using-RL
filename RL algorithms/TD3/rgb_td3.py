#!/usr/bin/env python

# Copyright (c) 2025: Mahdi Rahmani (mahdi.rahmani@uwaterloo.ca)

# Training code for TD3 agent in CARLA environment
# This code uses the RGB birdeye view as state input
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
import collections
import random
import time
from datetime import datetime
import cv2
import traceback
import sys
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', type=str, help='CARLA server host')
    parser.add_argument('--port', default=3000, type=int, help='CARLA server port')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic manager port')
    parser.add_argument('--sync', action='store_true', help='Synchronous mode')
    parser.add_argument('--no-rendering', action='store_true', help='No rendering mode')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of episodes to train')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to load model checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', default='./logs', type=str, help='Directory for tensorboard logs')
    parser.add_argument('--recovery-timeout', default=5, type=int, help='Time to wait after an error before retrying')
    parser.add_argument('--pedestrians', default=5, type=int, help='Number of pedestrians for jaywalking')
    
    # TD3 specific parameters 
    parser.add_argument('--policy-delay', default=3, type=int, help='Policy update frequency (TD3 delay parameter)')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate for critic')
    parser.add_argument('--actor-lr', default=1e-4, type=float, help='Learning rate for actor (lower than critic)')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--tau', default=0.005, type=float, help='Soft update coefficient')
    parser.add_argument('--policy-noise', default=0.4, type=float, help='Noise added to target policy')
    parser.add_argument('--noise-clip', default=0.5, type=float, help='Range to clip target policy noise')
    parser.add_argument('--exploration-noise', default=0.3, type=float, help='Std of Gaussian exploration noise')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Replay buffer size')
    parser.add_argument('--warmup-steps', default=2000, type=int, help='Initial random exploration steps')
    parser.add_argument('--action-reg', default=0.01, type=float, help='L2 regularization on actions')
    parser.add_argument('--extreme-penalty', default=0.05, type=float, help='Penalty for extreme actions')
    parser.add_argument('--rand-action-prob', default=0.15, type=float, help='Probability of random action')
    return parser.parse_args()

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        self.input_shape = input_shape  
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Layer normalization for improved stability
        self.norm1 = nn.LayerNorm([32, 20, 20])
        self.norm2 = nn.LayerNorm([64, 9, 9])
        self.norm3 = nn.LayerNorm([64, 7, 7])
        
        # Calculate feature size
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        self.feature_size = convw * convh * 64
        
        # Initialize weights using orthogonal initialization
        nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
    
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        return x.view(-1, self.feature_size)

class Actor(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        
        # Feature extractor
        self.features = CNNFeatureExtractor(input_shape)
        feature_size = self.features.feature_size
        
        # Actor network layers with dropouts for regularization
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.drop1 = nn.Dropout(0.2)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(0.2)  
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01) 
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        
        # Use tanh but scale to 80% of max to prevent saturation at extremes
        # This helps prevent the actions from getting stuck at the boundaries
        raw_actions = self.fc3(x)
        
        # Custom activation that biases toward center values
        actions = torch.tanh(raw_actions) * (self.max_action * 0.8)
        
        return actions

class Critic(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Feature extractor
        self.features = CNNFeatureExtractor(input_shape)
        feature_size = self.features.feature_size
        
        # First critic network (Q1)
        self.q1_fc1 = nn.Linear(feature_size + action_dim, hidden_dim)
        self.q1_drop1 = nn.Dropout(0.2)  
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_drop2 = nn.Dropout(0.2)  
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Second critic network (Q2)
        self.q2_fc1 = nn.Linear(feature_size + action_dim, hidden_dim)
        self.q2_drop1 = nn.Dropout(0.2)  
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_drop2 = nn.Dropout(0.2)  
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        # Better initialization
        nn.init.orthogonal_(self.q1_fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q1_fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q1_out.weight, gain=1.0)
        nn.init.constant_(self.q1_out.bias, 0)
        
        nn.init.orthogonal_(self.q2_fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q2_fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.q2_out.weight, gain=1.0)
        nn.init.constant_(self.q2_out.bias, 0)
    
    def forward(self, state, action):
        # Extract features from state
        state_features = self.features(state)
        
        # Concatenate state features and action
        x1 = torch.cat([state_features, action], dim=1)
        x1 = F.relu(self.q1_drop1(self.q1_fc1(x1)))
        x1 = F.relu(self.q1_drop2(self.q1_fc2(x1)))
        q1 = self.q1_out(x1)
        
        x2 = torch.cat([state_features, action], dim=1)
        x2 = F.relu(self.q2_drop1(self.q2_fc1(x2)))
        x2 = F.relu(self.q2_drop2(self.q2_fc2(x2)))
        q2 = self.q2_out(x2)
        
        return q1, q2
    
    def q1_value(self, state, action):
        """Return only the first Q-value for policy optimization"""
        state_features = self.features(state)
        x1 = torch.cat([state_features, action], dim=1)
        x1 = F.relu(self.q1_drop1(self.q1_fc1(x1)))
        x1 = F.relu(self.q1_drop2(self.q1_fc2(x1)))
        q1 = self.q1_out(x1)
        return q1

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.default_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.default_priority)  
    
    def sample(self, batch_size, beta=0.4):
        # If not enough samples, return random ones
        if len(self.buffer) < batch_size:
            indices = random.sample(range(len(self.buffer)), len(self.buffer))
        else:
            # Use priorities for sampling with probability proportional to priority
            priorities = np.array(self.priorities)
            probs = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones, indices
    
    def update_priorities(self, indices, priorities):
        # Update the priorities for these indices
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-5  
    
    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, state_shape, action_dim, args):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.args = args
        self.max_action = 1.0
        self.action_reg = args.action_reg  
        self.extreme_penalty = args.extreme_penalty  
        self.rand_action_prob = args.rand_action_prob  
        
        # Initialize networks
        self.actor = Actor(state_shape, action_dim, max_action=self.max_action).to(device)
        self.actor_target = Actor(state_shape, action_dim, max_action=self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_shape, action_dim).to(device)
        self.critic_target = Critic(state_shape, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers with different learning rates for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # TD3 hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_delay = args.policy_delay
        self.exploration_noise = args.exploration_noise
        self.warmup_steps = args.warmup_steps
        
        # Initialize replay buffer with priority replay
        self.memory = ReplayBuffer(args.buffer_size)
        
        # Initialize step counter
        self.total_steps = 0
        self.updates = 0
        
        # Action scaling (for environment interaction)
        self.action_high = torch.tensor([2.0, 0.6]).to(device)  
        self.action_low = torch.tensor([-0.5, -0.6]).to(device)
        self.action_range = self.action_high - self.action_low
        
        # For learning rate scheduling
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100000, gamma=0.5)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100000, gamma=0.5)
        
        self.actor.eval()
        self.critic.eval()
        
        # Noise process for exploration (Ornstein-Uhlenbeck for temporal correlation)
        self.noise_process = OUNoise(action_dim, sigma=self.exploration_noise)
    
    def select_action(self, state, evaluate=False):
        """Select action from policy with optional noise for exploration"""
        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        # During training, occasionally use pure random actions for exploration
        if not evaluate and (self.total_steps < self.warmup_steps or random.random() < self.rand_action_prob):
            # Pure exploration with uniform random but biased toward reasonable actions
            if random.random() < 0.7:  # 70% bias toward forward movement
                # Generate acceleration biased toward positive (forward)
                accel = random.uniform(-0.3, 1.0)  
                # Generate steering biased toward center
                steer = random.normalvariate(0, 0.3)  
                steer = max(-0.6, min(0.6, steer))  
                raw_action = np.array([accel, steer])
            else:
                # Complete random uniform
                raw_action = np.random.uniform(-0.9, 0.9, size=self.action_dim)
            
            # Scale to actual environment range
            action_scaled = self.action_low.cpu().numpy() + (raw_action + 1.0) * 0.5 * self.action_range.cpu().numpy()
            return np.clip(action_scaled, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
            
        # Disable dropout during action selection
        with torch.no_grad():
            # Get action from policy
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if not evaluate:
            # Add temporally correlated noise for better exploration
            noise = self.noise_process.sample()
            
            # Scale noise based on progress
            progress = min(1.0, self.total_steps / 200000)
            noise_scale = self.exploration_noise * (1.0 - 0.5 * progress)  
            
            # Apply noise with probabilistic mixing
            if random.random() < 0.8:  
                action = (action + noise * noise_scale).clip(-self.max_action, self.max_action)
        
        # Scale action from [-1,1] to actual environment ranges
        action_scaled = self.action_low.cpu().numpy() + (action + 1.0) * 0.5 * self.action_range.cpu().numpy()
        
        # Clip to ensure within bounds
        action_scaled = np.clip(action_scaled, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        # Apply a bias toward smoother steering when going straight
        if abs(action_scaled[1]) < 0.15 and random.random() < 0.4:
            action_scaled[1] *= 0.5  
            
        # Sometimes reduce extreme acceleration
        if action_scaled[0] > 1.7 and random.random() < 0.3:
            action_scaled[0] *= 0.8  
        
        return action_scaled
    
    def update(self, batch_size=128):
        """Update the networks using TD3 algorithm with improved stability"""
        if len(self.memory) < batch_size:
            return 0, 0  
        
        self.total_steps += 1
        
        # Set networks to training mode
        self.actor.train()
        self.critic.train()
        
        # Sample a batch of transitions with priorities
        states, actions, rewards, next_states, dones, indices = self.memory.sample(batch_size)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            
            # Add clipped noise for target policy smoothing 
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            
            # Get target Q values using the twin critics
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            
            # Take minimum to reduce overestimation bias
            target_q = torch.min(target_q1, target_q2)
            
            # Compute the target value with discount
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update priorities in replay buffer based on TD error
        with torch.no_grad():
            td_errors = torch.abs(target_q - current_q1).cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  
        self.critic_optimizer.step()
        
        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            self.updates += 1
            
            # Compute actor loss - maximize Q value
            actor_actions = self.actor(states)
            actor_q = self.critic.q1_value(states, actor_actions)
            
            # Add L2 regularization to penalize extreme actions
            action_l2_reg = self.action_reg * torch.mean(torch.square(actor_actions))
            
            # Add penalty for extreme actions (actions close to -1 or 1)
            # This encourages the network to output more moderate values
            extreme_action_penalty = self.extreme_penalty * torch.mean(torch.abs(torch.abs(actor_actions) - 0.5))
            
            # Minimize negative Q value (maximize Q) with regularization
            actor_loss = -actor_q.mean() + action_l2_reg + extreme_action_penalty
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  
            self.actor_optimizer.step()
            
            # Update target networks with soft update (Polyak averaging)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Step learning rate schedulers
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        # Set back to eval mode to ensure deterministic policy for action selection
        self.actor.eval()
        self.critic.eval()
        
        return critic_loss.item(), actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
    
    def save(self, path):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'total_steps': self.total_steps,
            'updates': self.updates
        }, path)
    
    def load(self, path):
        """Load model parameters"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if 'actor_scheduler' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler'])
        if 'critic_scheduler' in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
            
        self.total_steps = checkpoint['total_steps']
        self.updates = checkpoint.get('updates', 0)
        
        self.actor.eval()
        self.critic.eval()

# Ornstein-Uhlenbeck Noise process for temporally correlated exploration noise
class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated noise"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """Reset the internal state to mean"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """Update internal state and return it as noise"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# Preprocess the birdeye view observation
def preprocess_birdeye(birdeye):
    """Convert birdeye view image to model input format"""
    # Resize to 84x84
    resized = cv2.resize(birdeye, (84, 84))
    # Normalize pixel values
    normalized = resized / 255.0
    # Transpose for PyTorch 
    transposed = np.transpose(normalized, (2, 0, 1))  
    return transposed

# Exponential moving average tracker for rewards (for smoother visualization)
class EMARewardTracker:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.value = None
    
    def update(self, reward):
        if self.value is None:
            self.value = reward
        else:
            self.value = self.alpha * reward + (1 - self.alpha) * self.value
        return self.value

# Train a single episode
def train_single_episode(env, agent, writer, episode, args, reward_history, ema_tracker):
    """Train for a single episode with improved monitoring"""
    max_timesteps = 1000
    episode_reward = 0
    episode_steps = 0
    episode_critic_loss = 0
    episode_actor_loss = 0
    
    # Track action statistics for debugging
    actions_log = []
    
    # Reset environment
    print("Resetting environment...")
    sys.stdout.flush()
    obs, info = env.reset()
    
    # Reset noise process at the beginning of each episode
    agent.noise_process.reset()
    
    # Process initial observation
    state = preprocess_birdeye(obs['birdeye'])
    
    # Episode loop
    for t in range(max_timesteps):
        # Select action with noise for exploration
        action = agent.select_action(state)
        
        # Log action details periodically
        if t % 50 == 0:
            print(f"Step {t}, Action: {action}")
            sys.stdout.flush()
        
        # Track action statistics
        actions_log.append(action)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Process next observation
        next_state = preprocess_birdeye(next_obs['birdeye'])
        
        # Store transition in replay buffer (scaled to [-1, 1] for network)
        normalized_action = 2.0 * (action - agent.action_low.cpu().numpy()) / agent.action_range.cpu().numpy() - 1.0
        normalized_action = np.clip(normalized_action, -1.0, 1.0)  
        
        agent.memory.add(state, normalized_action, reward, next_state, float(done))
        
        # Update networks
        if len(agent.memory) >= args.batch_size:
            critic_loss, actor_loss = agent.update(args.batch_size)
            episode_critic_loss += critic_loss
            if isinstance(actor_loss, float):
                episode_actor_loss += actor_loss
        
        # Move to next state
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # Render
        env.render()
        
        # Check if done
        if done:
            if terminated:
                print(f"Episode terminated after {t+1} steps (collision/lane departure)")
            else:
                print(f"Episode truncated after {t+1} steps (time limit)")
            break
    
    # Convert actions log to numpy for statistics
    actions_np = np.array(actions_log)
    
    # Calculate action statistics if we have any actions
    if len(actions_log) > 0:
        accel_mean = np.mean(actions_np[:, 0])
        accel_std = np.std(actions_np[:, 0])
        steer_mean = np.mean(actions_np[:, 1])
        steer_std = np.std(actions_np[:, 1])
        
        # Calculate action histogram for throttle/brake
        accel_bins = np.linspace(-0.5, 2.0, 6)  # Updated for new max of 2.0
        accel_hist, _ = np.histogram(actions_np[:, 0], bins=accel_bins, density=True)
        
        # Calculate action histogram for steering
        steer_bins = np.linspace(-0.6, 0.6, 7)
        steer_hist, _ = np.histogram(actions_np[:, 1], bins=steer_bins, density=True)
        
        # Log detailed action statistics
        for i, (bin_val, count) in enumerate(zip(accel_bins[:-1], accel_hist)):
            writer.add_scalar(f'Actions/Accel_Bin_{bin_val:.1f}', count, episode)
        
        for i, (bin_val, count) in enumerate(zip(steer_bins[:-1], steer_hist)):
            writer.add_scalar(f'Actions/Steer_Bin_{bin_val:.1f}', count, episode)
    else:
        accel_mean = steer_mean = accel_std = steer_std = 0
    
    # Update EMA tracker
    ema_reward = ema_tracker.update(episode_reward)
    
    # Log metrics
    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Reward/EMA', ema_reward, episode)
    writer.add_scalar('Steps/Episode', episode_steps, episode)
    writer.add_scalar('Loss/Critic', episode_critic_loss / max(1, episode_steps), episode)
    writer.add_scalar('Loss/Actor', episode_actor_loss / max(1, episode_steps), episode)
    
    # Log action statistics
    writer.add_scalar('Actions/Accel_Mean', accel_mean, episode)
    writer.add_scalar('Actions/Accel_Std', accel_std, episode) 
    writer.add_scalar('Actions/Steer_Mean', steer_mean, episode)
    writer.add_scalar('Actions/Steer_Std', steer_std, episode)
    
    # Log agent parameters
    writer.add_scalar('Agent/ExplorationNoise', agent.exploration_noise, episode)
    writer.add_scalar('Agent/ActorLR', agent.actor_scheduler.get_last_lr()[0], episode)
    writer.add_scalar('Agent/CriticLR', agent.critic_scheduler.get_last_lr()[0], episode)
    writer.add_scalar('Agent/Updates', agent.updates, episode)
    writer.add_scalar('Agent/ReplayBufferSize', len(agent.memory), episode)
    
    # Add to reward history
    reward_history.append(episode_reward)
    
    # Print episode summary
    print(f"\nEpisode {episode+1} | Reward: {episode_reward:.2f} | EMA Reward: {ema_reward:.2f} | Steps: {episode_steps}")
    print(f"Critic Loss: {episode_critic_loss/max(1, episode_steps):.4f} | Actor Loss: {episode_actor_loss/max(1, episode_steps):.4f}")
    print(f"Action Stats - Accel Mean: {accel_mean:.2f}, Std: {accel_std:.2f} | Steer Mean: {steer_mean:.2f}, Std: {steer_std:.2f}")
    sys.stdout.flush()
    
    return episode_reward

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"td3_{timestamp}"))
    
    # Keep track of rewards
    reward_history = []
    ema_tracker = EMARewardTracker(alpha=0.1)
    
    # Create the gym environment
    env = None
    agent = None
    try:
        print("Creating CARLA environment...")
        sys.stdout.flush()
        
        # Main training loop
        for episode in range(args.episodes):
            try:
                print(f"\n--- Starting episode {episode+1}/{args.episodes} ---")
                sys.stdout.flush()
                
                # Clean up previous environment
                if env is not None:
                    print("Closing previous environment...")
                    try:
                        env.close()
                    except Exception as e:
                        print(f"Error closing environment: {e}")
                    env = None
                    time.sleep(args.recovery_timeout)
                
                # Environment parameters 
                params = {
                    'number_of_vehicles': 0,  
                    'number_of_walkers': 0,
                    'display_size': 256,
                    'max_past_step': 1,
                    'dt': 0.1,
                    'discrete': False,
                    'discrete_acc': [-0.5, 0.0, 2.0],  
                    'discrete_steer': [-0.6, 0.0, 0.6],
                    'continuous_accel_range': [-0.5, 2.0],  
                    'continuous_steer_range': [-0.6, 0.6],
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
                    'out_lane_thres': max(10.0, min(25.0, 25.0 - (500 - min(episode, 500)) * 0.02)),
                    'desired_speed': min(3.0 + episode * 0.01, 8.0),  
                    'max_ego_spawn_times': 200,
                    'display_route': True,
                    'pixor_size': 64,
                    'pixor': False,
                    'sync': args.sync,
                    'rendering': not args.no_rendering,
                    'jaywalking_pedestrians': episode > 100, 
                    'terminate_on_lane_departure': episode > 100,  
                    'waypoint_distance_threshold': max(5.0, 15.0 - episode * 0.02)
                }
                
                print(f"Creating new environment for episode {episode+1}...")
                sys.stdout.flush()
                env = gym.make('carla-v0', params=params)
                
                # State and action dimensions
                state_shape = (3, 84, 84)  
                action_dim = 2  
                
                # Create or load agent
                if episode == 0 or agent is None:
                    print("Creating TD3 agent...")
                    agent = TD3Agent(state_shape, action_dim, args)
                    if args.checkpoint:
                        print(f"Loading model from {args.checkpoint}")
                        agent.load(args.checkpoint)
                
                # Run episode
                print(f"Running episode {episode+1}...")
                sys.stdout.flush()
                
                try:
                    train_single_episode(env, agent, writer, episode, args, reward_history, ema_tracker)
                except RuntimeError as e:
                    print(f"Runtime error: {e}")
                    traceback.print_exc()
                    continue
                
                print(f"Episode {episode+1} completed successfully")
                sys.stdout.flush()
                
                # Save model periodically
                if episode % 5 == 0 or episode == args.episodes - 1:
                    save_path = os.path.join(args.save_dir, f"td3_model_episode_{episode+1}.pth")
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
        final_model_path = os.path.join(args.save_dir, "td3_final_model.pth")
        agent.save(final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
        
        # Plot and save reward history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(reward_history) + 1), reward_history, alpha=0.6, label='Episode Reward')
        
        # Calculate and plot moving average
        if len(reward_history) > 10:
            moving_avg = [np.mean(reward_history[max(0, i-10):i+1]) for i in range(len(reward_history))]
            plt.plot(range(1, len(reward_history) + 1), moving_avg, 'r-', label='10-Episode Moving Avg')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('TD3 Training Reward History')
        plt.legend()
        plt.grid(True)
        reward_plot_path = os.path.join(args.log_dir, 'td3_reward_history.png')
        plt.savefig(reward_plot_path)
        print(f"Reward history plot saved to {reward_plot_path}")
        
        # Save rewards to CSV
        import pandas as pd
        rewards_df = pd.DataFrame(reward_history, columns=['reward'])
        rewards_df.to_csv(os.path.join(args.log_dir, 'td3_reward_history.csv'))
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        traceback.print_exc()
    finally:
        # Final cleanup
        if env is not None:
            print("Closing CARLA environment...")
            env.close()
            print("CARLA environment closed")
        writer.close()
        print("Clean up completed.")

if __name__ == "__main__":
    main()