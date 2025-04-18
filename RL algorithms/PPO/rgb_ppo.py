#!/usr/bin/env python
# Copyright (c) 2025: Mahdi Rahmani (mahdi.rahmani@uwaterloo.ca)
# Training code for PPO agent in CARLA environment with RGB birdeye view
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
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument('--episodes', default=500, type=int, help='Number of episodes to train')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to load model checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', default='./logs', type=str, help='Directory for tensorboard logs')
    parser.add_argument('--recovery-timeout', default=5, type=int, help='Time to wait after an error before retrying')
    parser.add_argument('--pedestrians', default=0, type=int, help='Number of pedestrians for jaywalking')
    
    # PPO specific parameters
    parser.add_argument('--update-epochs', default=3, type=int, help='Number of update epochs')
    parser.add_argument('--mini-batch-size', default=64, type=int, help='Mini batch size for updates')
    parser.add_argument('--lr', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--gae-lambda', default=0.95, type=float, help='GAE lambda parameter')
    parser.add_argument('--clip-param', default=0.1, type=float, help='PPO clip parameter')
    parser.add_argument('--value-loss-coef', default=0.5, type=float, help='Value loss coefficient')
    parser.add_argument('--entropy-coef', default=0.03, type=float, help='Entropy coefficient')
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help='Maximum norm for gradient clipping')
    parser.add_argument('--steps-per-update', default=512, type=int, help='Number of steps to collect before update')
    
    return parser.parse_args()

class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape  # (3, 84, 84) for RGB
        
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        h = conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2)
        w = conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2)
        self.feature_size = h * w * 32
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(-1, self.feature_size)

class ActorNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        
        # Feature extractor
        self.features = SimpleCNN(input_shape)
        feature_size = self.features.feature_size
        
        # Actor network layers
        self.fc = nn.Linear(feature_size, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # Fixed log_std as learnable parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # Initialize slightly lower
        
        # initialization
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc(x))
        
        # Get mean of action distribution
        # Tanh to bound means between -1 and 1
        action_mean = torch.tanh(self.mean(x))  
        
        # Use parameter for log_std instead of network output
        log_std = self.log_std.expand(action_mean.size(0), -1)
        log_std = torch.clamp(log_std, -2.0, 0.0) 
        
        return action_mean, log_std
    
    def get_action(self, state, deterministic=False):
        action_mean, log_std = self.forward(state)
        
        if deterministic:
            return action_mean, None
        
        # Create normal distribution
        std = log_std.exp()
        normal = Normal(action_mean, std)
        
        # Sample action from the distribution
        x_t = normal.rsample() 
        
        # Calculate log probability and prevent extreme log probs
        log_prob = normal.log_prob(x_t)
        log_prob = torch.clamp(log_prob, -10, 2)  
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Clip actions to be between -1 and 1
        action = torch.clamp(x_t, -1, 1)
        
        return action, log_prob
    
    def evaluate_actions(self, state, action):
        action_mean, log_std = self.forward(state)
        
        # Create normal distribution
        std = log_std.exp()
        normal = Normal(action_mean, std)
        
        # Compute log probability of the action with stability clipping
        log_prob = normal.log_prob(action)
        log_prob = torch.clamp(log_prob, -10, 2)  
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Compute entropy of the action distribution
        entropy = normal.entropy().sum(dim=-1).mean()
        
        return log_prob, entropy

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        
        # Feature extractor
        self.features = SimpleCNN(input_shape)
        feature_size = self.features.feature_size
        
        # Critic network layers
        self.fc = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # initialization
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.constant_(self.value.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc(x))
        value = self.value(x)
        return value

class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.std = 1
        self.count = epsilon
        self.returns = []
        self.epsilon = epsilon

    def update(self, reward):
        self.returns.append(reward)
        if len(self.returns) > 100:  
            self.returns.pop(0)
        
        if len(self.returns) > 1:
            self.mean = np.mean(self.returns)
            self.std = np.std(self.returns) + self.epsilon
    
    def normalize(self, reward):
        self.update(reward)
        return (reward - self.mean) / (self.std + self.epsilon)

class RolloutBuffer:
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        
        # Initialize buffers for states, actions, rewards
        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32).to(device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.values = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.action_log_probs = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.advantages = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        self.returns = torch.zeros((capacity, 1), dtype=torch.float32).to(device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs, action, reward, done, value, action_log_prob):
        # Store transition in buffer
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.action_log_probs[self.ptr] = action_log_prob
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        # Compute GAE advantages and returns
        last_gae_lam = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
            
            # Compute delta = r + gamma * V(s') - V(s)
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            
            # Compute GAE advantage: A = delta + gamma * lambda * A'
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        # Compute returns: R = A + V
        self.returns = self.advantages + self.values
    
    def get_batches(self, batch_size):
        # Shuffle the buffer
        indices = torch.randperm(self.size)
        
        # Create batches
        batches = []
        for start_idx in range(0, self.size, batch_size):
            # Handle the case where batch_size doesn't divide self.size evenly
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.values[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices],
                self.action_log_probs[batch_indices]
            )
            batches.append(batch)
        
        return batches
    
    def clear(self):
        self.ptr = 0
        self.size = 0

class PPOAgent:
    def __init__(self, state_shape, action_dim, args):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.args = args
        
        # PPO hyperparameters
        self.steps_per_update = args.steps_per_update
        self.clip_param = args.clip_param
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.update_epochs = args.update_epochs
        self.mini_batch_size = args.mini_batch_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        
        # Initialize networks
        self.actor = ActorNetwork(state_shape, action_dim).to(device)
        self.critic = CriticNetwork(state_shape).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(args.steps_per_update, state_shape, action_dim)
        
        # Track steps for updates
        self.step_count = 0
        self.update_count = 0
        
        # Action space limits (-0.5 to 3.0 for acceleration)
        self.action_high = torch.tensor([3.0, 0.6]).to(device)
        self.action_low = torch.tensor([-0.5, -0.6]).to(device)
        self.action_range = self.action_high - self.action_low
        
        # For debugging
        self.debug_actions = []
        
        # Reward normalizer for stability
        self.reward_normalizer = RewardNormalizer()
        
        # Initial exploration settings
        self.exploration_phase = True
        self.exploration_steps = 2000
        
        # Learning rate scheduler
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.5)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.5)
    
    def select_action(self, state, evaluate=False):
        # Determine if we should use structured exploration
        exploration_factor = max(0.05, min(0.8, self.exploration_steps / (self.step_count + 500)))
        
        # During early training or with probability based on exploration factor, explore in a structured way
        if not evaluate and (self.step_count < 5000 or random.random() < exploration_factor):
            # Generate structured exploration actions
            if random.random() < 0.7:            # 70% go forward with small steering
                action_scaled = np.array([
                    random.uniform(-0.5, 3.0),  
                    random.uniform(-0.6, 0.6)    
                ])
            else:  # 30% moderate steering with moderate acceleration
                action_scaled = np.array([
                    random.uniform(0.0, 1.0), 
                    random.choice([-1, 1]) * random.uniform(0.3, 0.6)
                ])
            
            # Clip to valid ranges
            action_scaled = np.clip(action_scaled, 
                                   self.action_low.cpu().numpy(), 
                                   self.action_high.cpu().numpy())
            
            # Convert back to normalized form for buffer storage
            action = 2 * (action_scaled - self.action_low.cpu().numpy()) / self.action_range.cpu().numpy() - 1
            
            # For exploration actions, we don't need accurate log_probs or values
            with torch.no_grad():
                state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
                value = self.critic(state_tensor).cpu().numpy()[0]
                
            return action_scaled, action, 0.0, value
        
        # Regular policy-based action selection
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get action and log probability
            action, action_log_prob = self.actor.get_action(state_tensor, deterministic=evaluate)
            
            if self.step_count % 100 == 0:
                print(f"Debug - Raw action: {action.cpu().numpy()[0]}")
                
            value = self.critic(state_tensor)
        
        if action_log_prob is not None:
            self.debug_actions.append(action.cpu().numpy()[0])
            
        # Scale action from [-1,1] to actual ranges for environment
        action_np = action.cpu().numpy()[0]
        action_scaled = self.action_low.cpu().numpy() + (action_np + 1.0) * 0.5 * self.action_range.cpu().numpy()
        
        # Make sure action is within bounds
        action_scaled = np.clip(action_scaled, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        log_prob_np = action_log_prob.cpu().numpy()[0] if action_log_prob is not None else 0.0
        value_np = value.cpu().numpy()[0]
        
        return action_scaled, action_np, log_prob_np, value_np
    
    def update(self):
        # Check if we have collected enough steps for an update
        if self.rollout_buffer.size < self.mini_batch_size:
            return {}
        
        # Compute returns and advantages
        with torch.no_grad():
            last_obs = self.rollout_buffer.observations[self.rollout_buffer.size - 1].unsqueeze(0)
            last_value = self.critic(last_obs)
        
        self.rollout_buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Track metrics
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        
        # Normalize advantages
        advantages = self.rollout_buffer.advantages[:self.rollout_buffer.size]
        if advantages.size(0) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get all minibatches
        batches = self.rollout_buffer.get_batches(self.mini_batch_size)
        num_batches = len(batches)
        
        # Perform multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Process each batch
            for batch_idx, batch in enumerate(batches):
                # Unpack the batch
                obs_batch, action_batch, old_values, returns, advantages_batch, old_log_probs = batch
                
                # Skip nan values
                if torch.isnan(obs_batch).any() or torch.isnan(action_batch).any():
                    continue
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate_actions(obs_batch, action_batch)
                
                # Skip nan values
                if torch.isnan(new_log_probs).any():
                    continue
                
                # Get new value predictions
                values = self.critic(obs_batch)
                
                # Calculate policy ratio (π_new / π_old)
                ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -10, 2))
                
                # Clamp ratio for stability
                ratio = torch.clamp(ratio, 0.0, 10.0)
                
                # Compute surrogate losses
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                
                # PPO policy loss (negative for gradient ascent)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                value_loss1 = (values - returns).pow(2)
                value_loss2 = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Skip nan losses
                if torch.isnan(loss).any():
                    continue
                
                # Optimize actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Optimize critic
                self.critic_optimizer.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Track losses
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.item()
        
        # Update learning rate schedulers
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        
        # Update counts
        self.update_count += 1
        
        # Decay entropy coefficient (very slowly)
        if self.update_count % 10 == 0:
            self.entropy_coef = max(0.001, self.entropy_coef * 0.995)
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Calculate average metrics
        avg_value_loss = total_value_loss / (self.update_epochs * num_batches)
        avg_policy_loss = total_policy_loss / (self.update_epochs * num_batches)
        avg_entropy = total_entropy / (self.update_epochs * num_batches)
        
        # Print debug info
        print(f"Update #{self.update_count} complete - Value loss: {avg_value_loss:.4f}, Policy loss: {avg_policy_loss:.4f}, Entropy: {avg_entropy:.4f}, LR: {self.scheduler_actor.get_last_lr()[0]:.6f}")
        
        return {
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy
        }
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'update_count': self.update_count,
            'entropy_coef': self.entropy_coef,
            'actor_scheduler': self.scheduler_actor.state_dict(),
            'critic_scheduler': self.scheduler_critic.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint.get('update_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.args.entropy_coef)
        if 'actor_scheduler' in checkpoint:
            self.scheduler_actor.load_state_dict(checkpoint['actor_scheduler'])
        if 'critic_scheduler' in checkpoint:
            self.scheduler_critic.load_state_dict(checkpoint['critic_scheduler'])

# Preprocess the birdeye view observation
def preprocess_birdeye(birdeye):
    # Resize to a smaller size
    resized = cv2.resize(birdeye, (84, 84))
    # Normalize pixel values
    normalized = resized / 255.0
    # Transpose to get channels first (PyTorch format) (Shape will be (3, 84, 84))
    transposed = np.transpose(normalized, (2, 0, 1))  
    return transposed

# Train a single episode
def train_single_episode(env, agent, writer, episode, args, reward_history):
    max_timesteps = 1000
    episode_step = 0
    episode_reward = 0
    
    # Track metrics
    value_losses = []
    policy_losses = []
    entropies = []
    
    # For logging specific actions
    acc_values = []
    steer_values = []
    
    # Reset environment
    print("Resetting environment...")
    sys.stdout.flush()
    obs, info = env.reset()
    
    # Process observation
    state = preprocess_birdeye(obs['birdeye'])
    
    # Initialize episode reward normalizer
    episode_rewards = []
    
    # Episode loop
    for t in range(max_timesteps):
        # Select action
        action_scaled, action, action_log_prob, value = agent.select_action(state)
        
        # Log action details periodically
        if t % 50 == 0:
            print(f"Step {t}, Action: {action_scaled}")
            sys.stdout.flush()
        
        # Track action values for analysis
        acc_values.append(action_scaled[0])
        steer_values.append(action_scaled[1])
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_scaled)
        done = terminated or truncated
        
        # Apply reward penalties for early termination
        if done and t < 100:
            reward -= 10.0  # Severe penalty for very early termination
        
        # Save raw reward for history
        episode_reward += reward
        
        # Normalize reward
        episode_rewards.append(reward)
        normalized_reward = agent.reward_normalizer.normalize(reward)
        
        # Process next observation
        next_state = preprocess_birdeye(next_obs['birdeye'])
        
        # Convert to tensors for buffer
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor([normalized_reward], dtype=torch.float32).to(device)
        done_tensor = torch.tensor([float(done)], dtype=torch.float32).to(device)
        value_tensor = torch.tensor([value], dtype=torch.float32).to(device)
        log_prob_tensor = torch.tensor([action_log_prob], dtype=torch.float32).to(device)
        
        # Store in buffer
        agent.rollout_buffer.add(
            state_tensor, action_tensor, reward_tensor, done_tensor, value_tensor, log_prob_tensor
        )
        
        # Move to next state
        state = next_state
        episode_step += 1
        agent.step_count += 1
        
        # Render
        env.render()
        
        # Update PPO if enough steps collected
        if agent.rollout_buffer.size >= agent.steps_per_update:
            print(f"Updating PPO after {agent.rollout_buffer.size} steps...")
            update_metrics = agent.update()
            
            if update_metrics:
                value_losses.append(update_metrics['value_loss'])
                policy_losses.append(update_metrics['policy_loss'])
                entropies.append(update_metrics['entropy'])
        
        # Check if done
        if done:
            if terminated:
                print(f"Episode terminated after {t+1} steps (collision/lane departure)")
            else:
                print(f"Episode truncated after {t+1} steps (time limit)")
            break
    
    # Final update with remaining steps
    if agent.rollout_buffer.size > 0:
        update_metrics = agent.update()
        if update_metrics:
            value_losses.append(update_metrics['value_loss'])
            policy_losses.append(update_metrics['policy_loss'])
            entropies.append(update_metrics['entropy'])
    
    # Log episode metrics
    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Steps/Episode', episode_step, episode)
    
    if value_losses:
        avg_value_loss = np.mean(value_losses)
        writer.add_scalar('Loss/Value', avg_value_loss, episode)
    
    if policy_losses:
        avg_policy_loss = np.mean(policy_losses)
        writer.add_scalar('Loss/Policy', avg_policy_loss, episode)
    
    if entropies:
        avg_entropy = np.mean(entropies)
        writer.add_scalar('Loss/Entropy', avg_entropy, episode)
    
    # Action statistics
    if acc_values:
        writer.add_scalar('Action/Mean_Acceleration', np.mean(acc_values), episode)
        writer.add_scalar('Action/Std_Acceleration', np.std(acc_values), episode)
    
    if steer_values:
        writer.add_scalar('Action/Mean_Steering', np.mean(steer_values), episode)
        writer.add_scalar('Action/Std_Steering', np.std(steer_values), episode)
    
    # Add to reward history
    reward_history.append(episode_reward)
    
    # Print episode summary
    print(f"\nEpisode {episode+1} | Reward: {episode_reward:.2f} | Steps: {episode_step}")
    if value_losses:
        print(f"Value Loss: {np.mean(value_losses):.4f} | Policy Loss: {np.mean(policy_losses):.4f} | Entropy: {np.mean(entropies):.4f}")
    
    # Print action statistics
    if acc_values:
        print(f"Acceleration - Mean: {np.mean(acc_values):.2f}, Std: {np.std(acc_values):.2f}, Min: {np.min(acc_values):.2f}, Max: {np.max(acc_values):.2f}")
    if steer_values:
        print(f"Steering - Mean: {np.mean(steer_values):.2f}, Std: {np.std(steer_values):.2f}, Min: {np.min(steer_values):.2f}, Max: {np.max(steer_values):.2f}")
    
    sys.stdout.flush()
    
    return episode_reward

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    # Keep track of rewards
    reward_history = []
    
    # Create the gym environment
    env = None
    agent = None
    try:
        print("Creating CARLA environment...")
        sys.stdout.flush()
        
        # Training loop
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
                
                # Environment parameters with curriculum
                params = {
                    'number_of_vehicles': min(episode // 50, 10),  # Gradually introduce traffic
                    'number_of_walkers': args.pedestrians,
                    'display_size': 256,
                    'max_past_step': 1,
                    'dt': 0.1,
                    'discrete': False,
                    'discrete_acc': [-0.5, 0.0, 3.0],
                    'discrete_steer': [-0.6, 0.0, 0.6],
                    'continuous_accel_range': [-0.5, 3.0],  # Modified acceleration range
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
                    'out_lane_thres': max(5.0, min(15.0, 15.0 - (300 - episode) * 0.05)) if episode < 300 else 15.0,  # Gradually increase tolerance
                    'desired_speed': min(4.0 + episode * 0.02, 8.0),                                                  # Gradually increase target speed
                    'max_ego_spawn_times': 200,
                    'display_route': True,
                    'pixor_size': 64,
                    'pixor': False,
                    'sync': args.sync,
                    'rendering': not args.no_rendering,
                    'jaywalking_pedestrians': episode > 200,       # Introduce jaywalkers later
                    'terminate_on_lane_departure': episode > 150,  # Only terminate for lane departure after some learning
                }
                
                print(f"Creating new environment for episode {episode+1}...")
                sys.stdout.flush()
                env = gym.make('carla-v0', params=params)
                
                # State and action dimensions
                state_shape = (3, 84, 84)  
                action_dim = 2  
                
                # Create or load agent
                if episode == 0 or agent is None:
                    print("Creating PPO agent...")
                    agent = PPOAgent(state_shape, action_dim, args)
                    if args.checkpoint:
                        print(f"Loading model from {args.checkpoint}")
                        agent.load(args.checkpoint)
                
                # Run episode
                print(f"Running episode {episode+1}...")
                sys.stdout.flush()
                
                try:
                    train_single_episode(env, agent, writer, episode, args, reward_history)
                except RuntimeError as e:
                    print(f"Runtime error: {e}")
                    traceback.print_exc()
                    continue
                
                print(f"Episode {episode+1} completed successfully")
                sys.stdout.flush()
                
                # Save model periodically
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
        final_model_path = os.path.join(args.save_dir, "final_model.pth")
        agent.save(final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")
        
        # Plot and save reward history
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