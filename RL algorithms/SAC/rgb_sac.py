#!/usr/bin/env python

# Copyright (c) 2025: Mahdi Rahmani (mahdi.rahmani@uwaterloo.ca)
# Training code for SAC agent in CARLA environment
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
from torch.distributions import Normal
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
    parser.add_argument('--target-update', default=1, type=int, help='Target network update frequency (every N updates)')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--tau', default=0.005, type=float, help='Soft update coefficient')
    parser.add_argument('--alpha', default=0.2, type=float, help='Temperature parameter for entropy')
    parser.add_argument('--auto-alpha', action='store_true', help='Automatically tune alpha')
    parser.add_argument('--buffer-size', default=100000, type=int, help='Replay buffer size')
    return parser.parse_args()

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        self.input_shape = input_shape  # (3, 84, 84) for RGB
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
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

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extractor
        self.features = CNNFeatureExtractor(input_shape)
        feature_size = self.features.feature_size
        
        # Policy layers
        self.fc1 = nn.Linear(feature_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and log_std outputs for action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Sample action from normal distribution
        x_t = normal.rsample() 
        
        # Squash to [-1, 1]
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, accounting for the transformation
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for tanh
        # See the original SAC paper for details
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Feature extractor
        self.features = CNNFeatureExtractor(input_shape)
        feature_size = self.features.feature_size
        
        # Q1 network
        self.q1_fc1 = nn.Linear(feature_size + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2 network (for twin Q-learning)
        self.q2_fc1 = nn.Linear(feature_size + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        state_features = self.features(state)
        
        # Concatenate state features and action
        x1 = torch.cat([state_features, action], dim=1)
        x1 = F.relu(self.q1_fc1(x1))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        
        x2 = torch.cat([state_features, action], dim=1)
        x2 = F.relu(self.q2_fc1(x2))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        
        return q1, q2
    
    def q1(self, state, action):
        state_features = self.features(state)
        
        x1 = torch.cat([state_features, action], dim=1)
        x1 = F.relu(self.q1_fc1(x1))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        
        return q1

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_shape, action_dim, args):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.auto_alpha = args.auto_alpha
        self.batch_size = args.batch_size
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_shape, action_dim).to(device)
        self.q_net = QNetwork(state_shape, action_dim).to(device)
        self.target_q_net = QNetwork(state_shape, action_dim).to(device)
        
        # Initialize target network with policy network weights
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
        
        # Set target network to evaluation mode
        self.target_q_net.eval()
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        
        # If auto_alpha, initialize log_alpha and optimizer
        if self.auto_alpha:
            # Target entropy is -dim(A)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(args.buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
        self.updates_done = 0
        
        # For scaling our continuous actions
        self.action_scaling = torch.tensor([2.0, 0.6]).to(device)  # [acc_range, steer_range]
        self.action_bias = torch.tensor([1.5, 0.0]).to(device)     # No bias initially
    
    def select_action(self, state, evaluate=False):
        # Convert state to tensor and add batch dimension
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if evaluate:
                # Use mean action for evaluation (no exploration)
                mean, _ = self.policy_net(state)
                action = torch.tanh(mean)
            else:
                # Sample action for training (with exploration)
                action, _, _ = self.policy_net.sample(state)
        
        # Scale action from [-1,1] to actual ranges needed for the environment
        scaled_action = action * self.action_scaling + self.action_bias
        
        return scaled_action.cpu().numpy()[0]
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return 0, 0, 0  
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Update Q networks
        with torch.no_grad():
            # Sample actions from the policy for next states
            next_actions, next_log_probs, _ = self.policy_net.sample(next_states)
            
            # Get Q values from target network
            next_q1, next_q2 = self.target_q_net(next_states, next_actions)
            
            # Take the minimum of the two Q values (twin Q trick)
            next_q = torch.min(next_q1, next_q2)
            
            # Calculate target Q value with entropy term
            target_q = rewards + (1 - dones) * self.gamma * (next_q - self.alpha * next_log_probs)
        
        # Get current Q estimates
        current_q1, current_q2 = self.q_net(states, actions)
        
        # Calculate Q loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update Q networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update policy network (delayed)
        if self.updates_done % args.target_update == 0:
            # Freeze Q network to save computational effort during policy update
            for param in self.q_net.parameters():
                param.requires_grad = False
                
            # Sample actions from the policy
            pi, log_pi, _ = self.policy_net.sample(states)
            
            # Get Q values for the sampled actions
            q1, q2 = self.q_net(states, pi)
            q = torch.min(q1, q2)
            
            # Calculate policy loss with entropy term
            policy_loss = (self.alpha * log_pi - q).mean()
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Unfreeze Q network
            for param in self.q_net.parameters():
                param.requires_grad = True
                
            # Update alpha if auto_alpha is enabled
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
        else:
            policy_loss = torch.tensor(0.0)
            
        # Soft update target networks
        if self.updates_done % args.target_update == 0:
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.updates_done += 1
        
        return q_loss.item(), policy_loss.item(), self.alpha
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'alpha': self.alpha
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
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
    # Transpose to get channels first 
    transposed = np.transpose(normalized, (2, 0, 1)) 
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
    episode_q_loss = 0
    episode_policy_loss = 0
    episode_steps = 0
    
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
        agent.memory.add(state, action, reward, next_state, done)
        
        # Move to next state
        state = next_state
        
        # Update networks
        if len(agent.memory) >= args.batch_size:
            q_loss, policy_loss, alpha = agent.update()
            episode_q_loss += q_loss
            episode_policy_loss += policy_loss
        
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
    writer.add_scalar('Loss/Q_Loss', episode_q_loss / max(1, episode_steps), episode)
    writer.add_scalar('Loss/Policy_Loss', episode_policy_loss / max(1, episode_steps), episode)
    writer.add_scalar('Parameters/Alpha', agent.alpha, episode)
    writer.add_scalar('Steps/Episode', episode_steps, episode)
    
    # Add to reward history
    reward_history.append(episode_reward)
    
    print(f"\nEpisode {episode+1} | Reward: {episode_reward:.2f} | "
          f"Q Loss: {episode_q_loss/max(1, episode_steps):.4f} | "
          f"Policy Loss: {episode_policy_loss/max(1, episode_steps):.4f} | "
          f"Alpha: {agent.alpha:.4f} | Steps: {episode_steps}")
    sys.stdout.flush()
    
    return episode_reward

def main():
    global args
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    # Keep track of rewards for plotting
    reward_history = []
    
    # Parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,  
        'number_of_walkers': args.pedestrians,  
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
                state_shape = (3, 84, 84)  
                action_dim = 2  
                
                # Create agent (or load existing one)
                if episode == 0 or agent is None:
                    print("Creating SAC agent...")
                    agent = SACAgent(state_shape, action_dim, args)
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