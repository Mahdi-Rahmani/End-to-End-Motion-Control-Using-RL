#!/usr/bin/env python

# Testing code for saved PPO agent in CARLA environment
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
    parser = argparse.ArgumentParser(description='Test a trained PPO model in CARLA environment')
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
    
    return parser.parse_args()

# Simple CNN for feature extraction
class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # Calculate output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        h = conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2)
        w = conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2)
        self.feature_size = h * w * 32
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(-1, self.feature_size)

# Actor Network
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
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc(x))
        
        # Get mean of action distribution
        action_mean = torch.tanh(self.mean(x))
        
        # Use parameter for log_std
        log_std = self.log_std.expand(action_mean.size(0), -1)
        log_std = torch.clamp(log_std, -2.0, 0.0)
        
        return action_mean, log_std
    
    def get_action(self, state, deterministic=False):
        action_mean, log_std = self.forward(state)
        
        if deterministic:
            # For evaluation, just return the mean action
            return action_mean, None
        
        # Create normal distribution
        std = log_std.exp()
        normal = Normal(action_mean, std)
        
        # Sample action
        x_t = normal.rsample()
        
        # Clip actions to be between -1 and 1
        action = torch.clamp(x_t, -1, 1)
        
        return action, None

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        
        # Feature extractor
        self.features = SimpleCNN(input_shape)
        feature_size = self.features.feature_size
        
        # Critic network layers
        self.fc = nn.Linear(feature_size, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc(x))
        value = self.value(x)
        return value

# PPO Agent for testing
class PPOAgent:
    def __init__(self, state_shape, action_dim):
        self.state_shape = state_shape
        self.action_dim = action_dim
        
        # Initialize networks
        self.actor = ActorNetwork(state_shape, action_dim).to(device)
        self.critic = CriticNetwork(state_shape).to(device)
        
        # Action space limits (-0.5 to 3.0 for acceleration)
        self.action_high = torch.tensor([3.0, 0.6]).to(device)
        self.action_low = torch.tensor([-0.5, -0.6]).to(device)
        self.action_range = self.action_high - self.action_low
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")
    
    def select_action(self, state, evaluate=True):
        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            # Get action deterministically for evaluation
            action, _ = self.actor.get_action(state_tensor, deterministic=evaluate)
            
            # Get value estimate
            value = self.critic(state_tensor)
        
        # Scale action from [-1,1] to actual ranges
        action_np = action.cpu().numpy()[0]
        action_scaled = self.action_low.cpu().numpy() + (action_np + 1.0) * 0.5 * self.action_range.cpu().numpy()
        
        # Make sure action is within bounds
        action_scaled = np.clip(action_scaled, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        return action_scaled, value.cpu().numpy()[0]

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
    cv2.putText(display, f"Reward: {reward:.2f}", (10, 120), font, 0.7, (255, 255, 255), 2)
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
        # Select action deterministically
        action_scaled, _ = agent.select_action(state)
        
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
    
    if done and step < 1000:
        print(f"Episode terminated early (collision or lane departure)")
    
    # Return episode metrics
    return {
        'episode': episode_num,
        'reward': total_reward,
        'steps': step,
        'avg_speed': np.mean(speeds) if speeds else 0,
        'max_speed': np.max(speeds) if speeds else 0,
        'avg_acceleration': np.mean(accelerations) if accelerations else 0,
        'avg_abs_steering': np.mean(np.abs(steering_angles)) if steering_angles else 0,
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
        'continuous_accel_range': [-0.5, 3.0],
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
        agent = PPOAgent(state_shape, action_dim)
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
            
            # Print summary
            print("\n===== Test Results Summary =====")
            print(f"Total episodes: {len(results_df)}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Average speed: {avg_speed:.2f} km/h")
            print(f"Success rate: {success_rate:.2%}")
            
            # Save results to CSV
            csv_path = os.path.join(args.output_dir, 'test_results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Detailed results saved to {csv_path}")
            
            # Create plots
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.bar(range(len(results_df)), results_df['reward'])
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Rewards per Episode')
            
            plt.subplot(2, 2, 2)
            plt.bar(range(len(results_df)), results_df['steps'])
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('Episode Lengths')
            
            plt.subplot(2, 2, 3)
            plt.bar(range(len(results_df)), results_df['avg_speed'])
            plt.xlabel('Episode')
            plt.ylabel('Average Speed (km/h)')
            plt.title('Average Speed per Episode')
            
            plt.subplot(2, 2, 4)
            plt.bar(range(len(results_df)), results_df['avg_abs_steering'])
            plt.xlabel('Episode')
            plt.ylabel('Average Abs Steering')
            plt.title('Steering Magnitude per Episode')
            
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