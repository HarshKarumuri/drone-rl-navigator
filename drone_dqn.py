# AERO-RL: Deep Q-Network (DQN) Continuous Drone Navigator
# Concept: Replacing the Q-table spreadsheet with a Neural Network (Brain).
# Target Audience: Explained for a 10-year-old, written with full depth.

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import time

# --- STEP 1: UNDERSTAND CONTINUOUS STATE & SENSORS ---
# In a real drone, states aren't simple grid cells. The drone flies in continuous coordinates.
# To make this real, our drone state vector has 6 continuous measurements:
# State = [Drone_X, Drone_Y, Dist_Sensor_Up, Dist_Sensor_Down, Dist_Sensor_Left, Dist_Sensor_Right]
# The distance sensors mimic ultrasonic/laser sensors that report how far the closest wall or obstacle is.

GRID_SIZE = 10.0 # 10x10 meters coordinate system
START_POS = [1.0, 1.0]
GOAL_POS = [9.0, 9.0]
GOAL_THRESHOLD = 0.8 # Drone needs to get within 0.8 meters of the goal

# Static Obstacles (spheres with centers and radii)
# Obstacles: (x, y, radius)
OBSTACLES = [
    (3.0, 2.0, 0.8),
    (5.0, 5.0, 1.0),
    (7.0, 3.0, 0.8),
    (2.0, 7.0, 0.9),
    (8.0, 8.0, 0.7)
]

# Actions: 0=Up, 1=Down, 2=Left, 3=Right (Step size = 0.5 meters)
STEP_SIZE = 0.5

class DroneEnvContinuous:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.position = np.array(START_POS, dtype=np.float32)
        self.steps = 0
        return self._get_state()
        
    def _get_state(self):
        # Calculate sensor readings
        x, y = self.position
        
        # Distance to boundaries
        dist_up = y
        dist_down = GRID_SIZE - y
        dist_left = x
        dist_right = GRID_SIZE - x
        
        # Distance to obstacle surfaces in the 4 directions (simplified range-finding)
        for ox, oy, r in OBSTACLES:
            # Check vertical distance (Up/Down) if within horizontal radius
            if abs(x - ox) < r:
                if y > oy: # Obstacle is above
                    dist_up = min(dist_up, y - oy - r)
                else: # Obstacle is below
                    dist_down = min(dist_down, oy - y - r)
            # Check horizontal distance (Left/Right) if within vertical radius
            if abs(y - oy) < r:
                if x > ox: # Obstacle is left
                    dist_left = min(dist_left, x - ox - r)
                else: # Obstacle is right
                    dist_right = min(dist_right, ox - x - r)
                    
        # Ensure values are non-negative
        sensor_readings = np.clip([dist_up, dist_down, dist_left, dist_right], 0.0, GRID_SIZE)
        
        # Build 6D continuous state vector
        state = np.array([x, y, *sensor_readings], dtype=np.float32)
        return state
        
    def step(self, action):
        x, y = self.position
        self.steps += 1
        
        # 1. Apply movements
        if action == 0:    # Up
            y -= STEP_SIZE
        elif action == 1:  # Down
            y += STEP_SIZE
        elif action == 2:  # Left
            x -= STEP_SIZE
        elif action == 3:  # Right
            x += STEP_SIZE
            
        self.position = np.array([x, y], dtype=np.float32)
        
        # Check boundary crash
        if x <= 0.0 or x >= GRID_SIZE or y <= 0.0 or y >= GRID_SIZE:
            return self._get_state(), -30.0, True, "crash_boundary"
            
        # Check obstacle collision
        for ox, oy, r in OBSTACLES:
            dist = np.linalg.norm(self.position - np.array([ox, oy]))
            if dist < r:
                return self._get_state(), -40.0, True, "crash_obstacle"
                
        # Check goal reached
        dist_to_goal = np.linalg.norm(self.position - np.array(GOAL_POS))
        if dist_to_goal < GOAL_THRESHOLD:
            return self._get_state(), 100.0, True, "goal"
            
        # Normal step energy cost & distance reward
        # Give a small reward for moving closer to the target
        dist_before = np.linalg.norm(np.array([x, y]) - np.array(GOAL_POS))
        shaping = 0.5 * (dist_before - dist_to_goal) # positive if distance decreased
        
        reward = -1.0 + shaping
        return self._get_state(), reward, False, "fly"

# --- STEP 2: THE NEURAL NETWORK BRAIN (DQN) ---
# A Neural Network takes our 6D state inputs and predicts the Q-value (goodness score) for each of our 4 actions.
# We build a simple Multi-Layer Perceptron (MLP) with PyTorch.
class DQNBrain(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNBrain, self).__init__()
        # Two hidden layers of 64 neurons
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs 4 scores
        )
        
    def forward(self, x):
        return self.network(x)

# --- STEP 3: EXPERIENCE REPLAY BUFFER ---
# When training a network, we shouldn't learn only from the single step we just took.
# The drone might forget past events (like hitting an obstacle).
# So we save our steps in a memory bank (Replay Buffer) and randomly sample groups of memories (Batches) to train.
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.uint8))
                
    def __len__(self):
        return len(self.buffer)

# --- STEP 4: THE DEEP Q-LEARNING AGENT ---
class DQNAgent:
    def __init__(self, state_dim=6, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Two networks: Policy Network (learns every step) and Target Network (updates slowly to stabilize learning)
        self.policy_net = DQNBrain(state_dim, action_dim)
        self.target_net = DQNBrain(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is evaluation only
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory()
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.batch_size = 64
        self.target_update_frequency = 10 # Update target net every 10 episodes
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return torch.argmax(q_values).item()
                
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return # Wait until we have enough memories to train
            
        # Sample a batch of memories
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch Tensors
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # 1. Compute Q(s, a) using current policy network
        current_q = self.policy_net(states_t).gather(1, actions_t)
        
        # 2. Compute Target: Y = reward + gamma * max(Q_target(s', a')) * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
            
        # 3. Compute Loss (Difference between what we thought and what really happened)
        loss = nn.MSELoss()(current_q, target_q)
        
        # 4. Optimize the network (Gradient Descent)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon

# --- STEP 5: TRAINING ORCHESTRATION ---
def main():
    env = DroneEnvContinuous()
    agent = DQNAgent()
    
    total_episodes = 100
    print(f"Beginning Deep Q-Network Training (DQN). Total Episodes: {total_episodes}")
    print("Wait for memory buffer to fill and watch PyTorch backpropagate errors to optimize flight weights...")
    time.sleep(2)
    
    success_count = 0
    recent_rewards = deque(maxlen=20)
    
    for ep in range(1, total_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            steps += 1
            action = agent.select_action(state)
            next_state, reward, done, status = env.step(action)
            
            # Save experience in replay memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the network!
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if done:
                if status == "goal":
                    success_count += 1
                break
                
            if steps > 100: # Limit steps per episode
                break
                
        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)
        
        # Update Target network weights periodically
        if ep % agent.target_update_frequency == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if ep % 5 == 0 or ep == 1:
            print(f"Episode: {ep:3d}/{total_episodes} | Steps: {steps:3d} | Ep Reward: {total_reward:6.1f} | Avg Reward (last 20): {avg_reward:6.1f} | Epsilon: {agent.epsilon:.2f} | Buffer Size: {len(agent.memory)}")
            
    # Save the trained brain weights!
    model_path = "drone_dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"\nDQN training finished! Trained policy saved to: {model_path}")
    print("In a real deployment at ND Matrix, you would load these weight files (.pth) onto the drone controller hardware (like Raspberry Pi or NVIDIA Jetson) to execute autonomous path planning.")

if __name__ == "__main__":
    main()
