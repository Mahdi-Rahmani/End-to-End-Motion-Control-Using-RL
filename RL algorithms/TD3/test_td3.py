#!/usr/bin/env python

# Copyright (c) 2025: Mahdi Rahmani (mahdi.rahmani@uwaterloo.ca)

# Testing code for saved TD3 agent in CARLA environment
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
import random
import time
from datetime import datetime
import cv2
import traceback
import sys
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained TD3 model in CARLA environment')
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
        
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        self.feature_size = convw * convh * 64
    
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
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        
        # Use tanh but scale to 80% of max to prevent saturation at extremes
        raw_actions = self.fc3(x)
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

class TD3TestAgent:
    def __init__(self, state_shape, action_dim, hidden_dim=256):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.max_action = 1.0
        
        # Initialize actor network
        self.actor = Actor(state_shape, action_dim, hidden_dim, self.max_action).to(device)
        
        # Initialize critic network (optional for testing)
        self.critic = Critic(state_shape, action_dim, hidden_dim).to(device)
        
        # Action scaling
        self.action_high = torch.tensor([2.0, 0.6], device=device)  
        self.action_low = torch.tensor([-0.5, -0.6], device=device)
        self.action_range = self.action_high - self.action_low
        
        self.actor.eval()
        self.critic.eval()
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        
        # Load actor network
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            print("Actor network loaded successfully")
        else:
            print("Warning: Actor network not found in checkpoint")
        
        # Load critic network (optional for testing)
        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
            print("Critic network loaded successfully")
        else:
            print("Warning: Critic network not found in checkpoint")
        
        print(f"Model loaded from {path}")
    
    def select_action(self, state, deterministic=True):
        """
        Select action from the policy.
        During testing, we always use deterministic actions.
        """
        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        
        # Get deterministic action from policy
        with torch.no_grad():
            # Turn off dropout during evaluation
            self.actor.eval()
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Scale action from [-1,1] to actual environment ranges
        action_scaled = self.action_low.cpu().numpy() + (action + 1.0) * 0.5 * self.action_range.cpu().numpy()
        
        # Clip to ensure within bounds
        action_scaled = np.clip(action_scaled, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        
        return action_scaled

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
    resized = cv2.resize(birdeye, (84, 84))
    normalized = resized / 255.0
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
    step_times = []
    action_history = []
    
    # Episode loop
    start_time = time.time()
    while not done and step < 1000:  
        step_start = time.time()
        
        # Select action deterministically for evaluation
        action_scaled = agent.select_action(state)
        
        # Get action components
        acceleration = action_scaled[0]
        steering = action_scaled[1]
        
        # Track actions
        action_history.append(action_scaled)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_scaled)
        done = terminated or truncated
        
        # Process next observation
        next_state = preprocess_birdeye(next_obs['birdeye'])
        
        # Get vehicle speed in km/h
        if hasattr(env, 'ego'):
            v = env.ego.get_velocity()
            speed = 3.6 * np.sqrt(v.x**2 + v.y**2)  
        else:
            speed = 0
        
        # Record frame if requested
        if recorder:
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
        step_times.append(time.time() - step_start)
        
        # Update state and reward
        state = next_state
        total_reward += reward
        step += 1
        
        # Log progress periodically
        if step % 10 == 0:
            avg_fps = 10 / max(0.001, sum(step_times[-10:]))
            print(f"Step {step}, Reward: {total_reward:.2f}, Speed: {speed:.1f} km/h, FPS: {avg_fps:.1f}")
        
        # Render
        env.render()
    
    if recorder:
        recorder.save()
    
    # Calculate metrics
    episode_duration = time.time() - start_time
    avg_fps = step / max(0.001, episode_duration)
    
    # Convert action history to numpy for analysis
    actions_np = np.array(action_history)
    
    # Print results
    print(f"\nEpisode {episode_num} completed:")
    print(f"Steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Duration: {episode_duration:.1f} seconds")
    print(f"Average FPS: {avg_fps:.1f}")
    
    if speeds:
        avg_speed = np.mean(speeds)
        print(f"Average speed: {avg_speed:.2f} km/h")
    
    # Additional metrics
    success = False
    if step >= 1000:  
        success = True
        print(f"Episode completed successfully (reached max steps)")
    else:
        print(f"Episode terminated early (collision or lane departure)")
    
    # Action smoothness metric 
    if len(steering_angles) > 1:
        steering_smoothness = np.mean(np.abs(np.diff(steering_angles)))
        accel_smoothness = np.mean(np.abs(np.diff(accelerations)))
    else:
        steering_smoothness = 0
        accel_smoothness = 0
    
    # Action distribution metrics
    if len(actions_np) > 0:
        accel_bins = np.linspace(-0.5, 2.0, 6)
        accel_hist, _ = np.histogram(actions_np[:, 0], bins=accel_bins, density=True)
        
        steer_bins = np.linspace(-0.6, 0.6, 7)
        steer_hist, _ = np.histogram(actions_np[:, 1], bins=steer_bins, density=True)
        
        print("Acceleration distribution:")
        for i, (low, high) in enumerate(zip(accel_bins[:-1], accel_bins[1:])):
            print(f"  {low:.1f} to {high:.1f}: {accel_hist[i]*100:.1f}%")
        
        print("Steering distribution:")
        for i, (low, high) in enumerate(zip(steer_bins[:-1], steer_bins[1:])):
            print(f"  {low:.1f} to {high:.1f}: {steer_hist[i]*100:.1f}%")
    
    # Return episode metrics
    return {
        'episode': episode_num,
        'reward': total_reward,
        'steps': step,
        'success': success,
        'avg_speed': np.mean(speeds) if speeds else 0,
        'max_speed': np.max(speeds) if speeds else 0,
        'avg_acceleration': np.mean(accelerations) if accelerations else 0,
        'avg_abs_steering': np.mean(np.abs(steering_angles)) if steering_angles else 0,
        'steering_smoothness': steering_smoothness,
        'accel_smoothness': accel_smoothness,
        'action_excess': np.mean(np.abs(actions_np > 0.9 * agent.action_high.cpu().numpy()).astype(float)) if len(actions_np) > 0 else 0,
        'accel_histogram': accel_hist.tolist() if len(actions_np) > 0 else [],
        'steer_histogram': steer_hist.tolist() if len(actions_np) > 0 else [],
        'duration': episode_duration,
        'fps': avg_fps,
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
        state_shape = (3, 84, 84)  
        action_dim = 2 
        
        # Create agent and load model
        agent = TD3TestAgent(state_shape, action_dim, hidden_dim=256)
        agent.load(args.model_path)
        
        # Run test episodes
        for episode in range(args.episodes):
            try:
                # Create new environment for each episode
                if env is not None:
                    env.close()
                    time.sleep(3)  
                
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
            # Create a copy of results without histograms for CSV
            results_for_csv = []
            for result in all_results:
                result_copy = result.copy()
                # Remove histogram data as it's not suitable for CSV
                result_copy.pop('accel_histogram', None)
                result_copy.pop('steer_histogram', None)
                results_for_csv.append(result_copy)
                
            # Convert to DataFrame
            results_df = pd.DataFrame(results_for_csv)
            
            # Calculate aggregate statistics
            avg_reward = results_df['reward'].mean()
            avg_steps = results_df['steps'].mean()
            avg_speed = results_df['avg_speed'].mean()
            success_rate = results_df['success'].mean() * 100
            steering_smoothness = results_df['steering_smoothness'].mean()
            accel_smoothness = results_df['accel_smoothness'].mean()
            avg_fps = results_df['fps'].mean()
            
            # Print summary
            print("\n===== Test Results Summary =====")
            print(f"Total episodes: {len(results_df)}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average steps: {avg_steps:.2f}")
            print(f"Average speed: {avg_speed:.2f} km/h")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Steering smoothness: {steering_smoothness:.4f} (lower is better)")
            print(f"Acceleration smoothness: {accel_smoothness:.4f} (lower is better)")
            print(f"Average FPS: {avg_fps:.1f}")
            
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
            
            plt.subplot(2, 3, 6)
            plt.bar(range(len(results_df)), results_df['action_excess'])
            plt.xlabel('Episode')
            plt.ylabel('Action Excess Rate')
            plt.title('Frequency of Extreme Actions')
            
            plt.tight_layout()
            
            # Save plot
            plt_path = os.path.join(args.output_dir, 'test_plots.png')
            plt.savefig(plt_path)
            print(f"Result plots saved to {plt_path}")
            
            # Create action distribution plot
            if len(all_results) > 0 and 'accel_histogram' in all_results[0] and 'steer_histogram' in all_results[0]:
                # Average the histograms across episodes
                avg_accel_hist = np.mean([result['accel_histogram'] for result in all_results], axis=0)
                avg_steer_hist = np.mean([result['steer_histogram'] for result in all_results], axis=0)
                
                plt.figure(figsize=(12, 5))
                
                # Acceleration distribution
                plt.subplot(1, 2, 1)
                accel_bins = np.linspace(-0.5, 2.0, 6)
                plt.bar((accel_bins[:-1] + accel_bins[1:]) / 2, avg_accel_hist, width=0.3)
                plt.xlabel('Acceleration')
                plt.ylabel('Frequency')
                plt.title('Acceleration Distribution')
                plt.grid(True, alpha=0.3)
                
                # Steering distribution
                plt.subplot(1, 2, 2)
                steer_bins = np.linspace(-0.6, 0.6, 7)
                plt.bar((steer_bins[:-1] + steer_bins[1:]) / 2, avg_steer_hist, width=0.15)
                plt.xlabel('Steering')
                plt.ylabel('Frequency')
                plt.title('Steering Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                action_dist_path = os.path.join(args.output_dir, 'action_distribution.png')
                plt.savefig(action_dist_path)
                print(f"Action distribution plot saved to {action_dist_path}")
    
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