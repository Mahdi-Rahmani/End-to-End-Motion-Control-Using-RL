#!/usr/bin/env python

# Testing code for saved T-SAC agent in CARLA environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gym
import gym_carla
import carla
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
from datetime import datetime
import cv2
import traceback
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained T-SAC model in CARLA environment')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--host', default='localhost', type=str, help='CARLA server host')
    parser.add_argument('--port', default=3000, type=int, help='CARLA server port')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic manager port')
    parser.add_argument('--episodes', default=10, type=int, help='Number of episodes to test')
    parser.add_argument('--sync', action='store_true', help='Synchronous mode')
    parser.add_argument('--no-rendering', action='store_true', help='No rendering mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--record', action='store_true', help='Record video of episodes')
    parser.add_argument('--vehicles', type=int, default=10, help='Number of vehicles')
    parser.add_argument('--pedestrians', type=int, default=5, help='Number of pedestrians')
    parser.add_argument('--output-dir', default='./test_results', help='Directory for test results')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for networks')
    parser.add_argument('--max-length', type=int, default=8, help='Maximum sequence length for transformer')
    
    return parser.parse_args()

# CNN feature extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNFeatureExtractor, self).__init__()
        self.input_shape = input_shape  # (3, 84, 84) for RGB
        
        # CNN layers
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
        
    def forward(self, x):
        features = self.features(x)
        
        # Mean network
        mean_x = F.leaky_relu(self.mean_ln1(self.mean_fc1(features)))
        mean_x = F.leaky_relu(self.mean_ln2(self.mean_fc2(mean_x)))
        mean = self.mean_out(mean_x)
        
        # Variance network (with mean input)
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

# Transformer-based Critic Network 
class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, max_seq_len=8):
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
        
        # Simplified transformer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim*2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=1
        )
        
        # Output heads for different sequence lengths
        self.q_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(max_seq_len)])
    
    def forward(self, state, actions):
        """
        Args:
            state: (batch_size, state_dim) tensor
            actions: (batch_size, seq_len, action_dim) tensor
        
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
        
        # Causal mask for transformer
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(actions.device)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(combined, mask=mask)
        
        # Get Q-values for each subsequence length
        q_values = []
        for i in range(min(seq_len, len(self.q_outputs))):
            q_i = self.q_outputs[i](transformer_output[:, i])
            q_values.append(q_i)
        
        return q_values

# T-SAC Agent for testing
class TSACTestAgent:
    def __init__(self, state_shape, action_dim, hidden_dim=128, max_seq_len=8):
        self.state_shape = state_shape
        self.action_dim = action_dim
        
        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor(state_shape).to(device)
        feature_size = self.feature_extractor.feature_size
        
        # Initialize policy network
        self.policy_net = TSACPolicyNetwork(
            state_shape, 
            action_dim, 
            hidden_dim=hidden_dim
        ).to(device)
        
        # Initialize critic for value estimation during testing (optional)
        self.critic = TransformerCritic(
            state_dim=feature_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len
        ).to(device)
        
        # For scaling our continuous actions
        self.action_scaling = torch.tensor([2.0, 0.6], device=device)  # [acc_range, steer_range]
        self.action_bias = torch.tensor([1.5, 0.0], device=device)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        
        # Load policy network
        if 'policy_net' in checkpoint:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            print("Policy network loaded successfully")
        else:
            print("Warning: Policy network not found in checkpoint")
        
        # Load critic network (optional for testing)
        if 'critic1' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic1'])
            print("Critic network loaded successfully")
        else:
            print("Warning: Critic network not found in checkpoint")
        
        print(f"Model loaded from {path}")
    
    def select_action(self, state, deterministic=True):
        # Convert state to tensor
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if deterministic:
                # Use mean action for evaluation
                mean, _ = self.policy_net(state_tensor)
                action = torch.tanh(mean)
            else:
                # Sample action for exploration
                action, _, _ = self.policy_net.sample(state_tensor)
        
        # Scale action from [-1,1] to actual ranges needed for the environment
        scaled_action = action * self.action_scaling + self.action_bias
        
        return scaled_action.cpu().numpy()[0]

# Video recorder class
class VideoRecorder:
    def __init__(self, output_dir, episode_num):
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f'episode_{episode_num}.mp4')
        self.frames = []
        
    def add_frame(self, frame):
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frames.append(frame)
    
    def save(self):
        if not self.frames:
            print("No frames to save!")
            return
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, 10, (width, height))
        
        # Write frames
        for frame in self.frames:
            out.write(frame)
        
        # Release video writer
        out.release()
        print(f"Video saved to {self.output_file}")

# Preprocess birdeye view
def preprocess_birdeye(birdeye):
    # Resize to network input size
    resized = cv2.resize(birdeye, (84, 84))
    # Normalize pixel values
    normalized = resized / 255.0
    # Transpose to get channels first (PyTorch format)
    transposed = np.transpose(normalized, (2, 0, 1))
    return transposed

# Add telemetry overlay to frame
def add_telemetry(frame, speed, acceleration, steering, reward, step):
    # Create a copy to avoid modifying original
    display = frame.copy()
    
    # Add text overlay with telemetry
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, f"Speed: {speed:.1f} km/h", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Accel: {acceleration:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Steer: {steering:.2f}", (10, 90), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Reward: {reward*2+1:.2f}", (10, 120), font, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Step: {step}", (10, 150), font, 0.7, (255, 255, 255), 2)
    
    return display

# Test function for a single episode
def test_episode(env, agent, episode_num, args):
    # Initialize recorder if needed
    recorder = None
    if args.record:
        recorder = VideoRecorder(os.path.join(args.output_dir, 'videos'), episode_num)
    
    # Reset environment
    print(f"Starting test episode {episode_num}...")
    obs, _ = env.reset()
    
    # Process observation
    state = preprocess_birdeye(obs['birdeye'])
    
    # Episode data
    total_reward = 0
    step = 0
    done = False
    
    # Track metrics
    speeds = []
    accelerations = []
    steering_angles = []
    rewards = []
    
    # Episode loop
    while not done and step < 1000:  # Max 1000 steps per episode
        # Select action deterministically for evaluation
        action_scaled = agent.select_action(state, deterministic=True)
        
        # Get action components
        acceleration = action_scaled[0]
        steering = action_scaled[1]
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_scaled)
        done = terminated or truncated
        
        # Process next observation
        next_state = preprocess_birdeye(next_obs['birdeye'])
        
        # Get vehicle speed in km/h
        if hasattr(env, 'ego'):
            v = env.ego.get_velocity()
            speed = 3.6 * np.sqrt(v.x**2 + v.y**2)  # Convert to km/h
        else:
            speed = 0
        
        # Record frame if requested
        if recorder:
            # Add telemetry overlay to frame
            display_img = add_telemetry(
                next_obs['birdeye'],
                speed,
                acceleration,
                steering,
                reward,
                step
            )
            recorder.add_frame(display_img)
        
        # Track metrics
        speeds.append(speed)
        accelerations.append(acceleration)
        steering_angles.append(steering)
        rewards.append(reward)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        step += 1
        
        # Log progress periodically
        if step % 10 == 0:
            print(f"Step {step}, Reward: {total_reward:.2f}, Speed: {speed:.1f} km/h")
        
        # Render
        env.render()
    
    # Save video if recorded
    if recorder:
        recorder.save()
    
    # Print results
    print(f"\nEpisode {episode_num} completed:")
    print(f"Steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    
    if speeds:
        avg_speed = np.mean(speeds)
        print(f"Average speed: {avg_speed:.2f} km/h")
    
    # Additional metrics specific to this episode
    lanekeeping_rate = 1.0
    if done and step < 1000:
        print(f"Episode terminated early (collision or lane departure)")
        lanekeeping_rate = 0.0
    
    # Action smoothness metric (lower is better)
    if len(steering_angles) > 1:
        steering_smoothness = np.mean(np.abs(np.diff(steering_angles)))
    else:
        steering_smoothness = 0
    
    # Return episode metrics
    return {
        'episode': episode_num,
        'reward': total_reward,
        'steps': step,
        'avg_speed': np.mean(speeds) if speeds else 0,
        'max_speed': np.max(speeds) if speeds else 0,
        'avg_acceleration': np.mean(accelerations) if accelerations else 0,
        'avg_abs_steering': np.mean(np.abs(steering_angles)) if steering_angles else 0,
        'steering_smoothness': steering_smoothness,
        'lanekeeping_rate': lanekeeping_rate,
        'early_termination': done and step < 1000
    }

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parameters for CARLA environment
    params = {
        'number_of_vehicles': args.vehicles,
        'number_of_walkers': args.pedestrians,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-0.5, 0.0, 3.0],
        'discrete_steer': [-0.6, 0.0, 0.6],
        'continuous_accel_range': [-3.0, 3.0],
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
    
    # Track all episode results
    all_results = []
    
    # Create environment and agent
    env = None
    try:
        # Initialize state dimensions
        state_shape = (3, 84, 84)  # RGB image (C, H, W)
        action_dim = 2  # [throttle/brake, steering]
        
        # Create agent and load model
        agent = TSACTestAgent(
            state_shape, 
            action_dim, 
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_length
        )
        agent.load(args.model_path)
        
        # Run test episodes
        for episode in range(args.episodes):
            try:
                # Create new environment for each episode
                if env is not None:
                    env.close()
                    time.sleep(3)  # Give CARLA time to clean up
                
                print(f"\nCreating environment for test episode {episode+1}/{args.episodes}...")
                env = gym.make('carla-v0', params=params)
                
                # Run episode and collect results
                episode_result = test_episode(env, agent, episode+1, args)
                all_results.append(episode_result)
                
                print(f"Episode {episode+1} results: Reward = {episode_result['reward']:.2f}, Steps = {episode_result['steps']}")
                
            except Exception as e:
                print(f"Error in episode {episode+1}: {e}")
                traceback.print_exc()
        
        # Analyze and save results
        if all_results:
            # Convert to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Calculate aggregate statistics
            avg_reward = results_df['reward'].mean()
            avg_steps = results_df['steps'].mean()
            avg_speed = results_df['avg_speed'].mean()
            success_rate = 1.0 - (results_df['early_termination'].sum() / len(results_df))
            steering_smoothness = results_df['steering_smoothness'].mean()
            
            # Print summary
            print("\n===== Test Results Summary =====")
            print(f"Total episodes: {len(results_df)}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Average speed: {avg_speed:.2f} km/h")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Steering smoothness: {steering_smoothness:.4f}")
            
            # Save results to CSV
            csv_path = os.path.join(args.output_dir, 'test_results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Detailed results saved to {csv_path}")
            
            # Create plots
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.bar(range(len(results_df)), results_df['reward'])
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Rewards per Episode')
            
            plt.subplot(2, 3, 2)
            plt.bar(range(len(results_df)), results_df['steps'])
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('Episode Lengths')
            
            plt.subplot(2, 3, 3)
            plt.bar(range(len(results_df)), results_df['avg_speed'])
            plt.xlabel('Episode')
            plt.ylabel('Average Speed (km/h)')
            plt.title('Average Speed per Episode')
            
            plt.subplot(2, 3, 4)
            plt.bar(range(len(results_df)), results_df['avg_abs_steering'])
            plt.xlabel('Episode')
            plt.ylabel('Average Abs Steering')
            plt.title('Steering Magnitude per Episode')
            
            plt.subplot(2, 3, 5)
            plt.bar(range(len(results_df)), results_df['steering_smoothness'])
            plt.xlabel('Episode')
            plt.ylabel('Steering Smoothness')
            plt.title('Steering Smoothness (lower is better)')
            
            plt.tight_layout()
            
            # Save plot
            plt_path = os.path.join(args.output_dir, 'test_plots.png')
            plt.savefig(plt_path)
            print(f"Result plots saved to {plt_path}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        traceback.print_exc()
    finally:
        # Final cleanup
        if env is not None:
            env.close()
        print("Testing completed.")

if __name__ == "__main__":
    main()