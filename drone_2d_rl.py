# AERO-RL: 2D Drone Obstacle-Avoidance Navigator
# Concept: A drone in a 2D grid world learning to find a landing pad avoiding static obstacles.
# Target Audience: Explained for a 10-year-old, written with full depth.

import random
import time
import os
import sys

# Ensure UTF-8 output on Windows consoles to support drone emojis
sys.stdout.reconfigure(encoding='utf-8')

GRID_SIZE = 10
START_POS = (0, 0)
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "UP (↑)", 1: "DOWN (↓)", 2: "LEFT (←)", 3: "RIGHT (→)"}

# Static Obstacles (same layout as the browser simulator)
OBSTACLES = {
    (2, 0), (2, 1), (2, 2),
    (4, 4), (4, 5), (4, 6), (4, 7),
    (7, 2), (7, 3), (7, 4),
    (1, 6), (2, 6), (3, 6),
    (7, 7), (8, 7), (6, 8)
}

# Hyperparameters
ALPHA = 0.15  # Learning rate
GAMMA = 0.90  # Discount factor
EPSILON = 1.0  # Starts at 1.0 (pure exploration)
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.05

# Initialize Q-Table: state (x,y) -> [q_up, q_down, q_left, q_right]
q_table = {}
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        q_table[(x, y)] = [0.0, 0.0, 0.0, 0.0]

def step(state, action):
    """
    Takes an action from the current state and returns:
    (next_state, reward, done, status_msg)
    """
    x, y = state
    
    if action == 0:    # Up
        next_y = y - 1
        next_x = x
    elif action == 1:  # Down
        next_y = y + 1
        next_x = x
    elif action == 2:  # Left
        next_x = x - 1
        next_y = y
    elif action == 3:  # Right
        next_x = x + 1
        next_y = y

    # 1. Out of Bounds Check
    if next_x < 0 or next_x >= GRID_SIZE or next_y < 0 or next_y >= GRID_SIZE:
        return state, -10.0, False, "Hit Boundary Guardrail! Stayed in place."

    next_state = (next_x, next_y)

    # 2. Obstacle Collision Check
    if next_state in OBSTACLES:
        return START_POS, -40.0, True, f"COLLISION at ({next_x}, {next_y})! Reset to start."

    # 3. Target Reached Check
    if next_state == GOAL_POS:
        return next_state, 100.0, True, "GOAL ARRIVAL! Success!"

    # 4. Cruising Penalty & Shaping Bonus
    # Calculate Manhattan distance to goal
    dist_before = abs(x - GOAL_POS[0]) + abs(y - GOAL_POS[1])
    dist_after = abs(next_x - GOAL_POS[0]) + abs(next_y - GOAL_POS[1])
    shaping = 0.2 if dist_after < dist_before else -0.2
    
    return next_state, -1.0 + shaping, False, "Safe Cruise."

def select_action(state, eps):
    """
    Epsilon-greedy selection
    """
    if random.random() < eps:
        return random.choice(ACTIONS)
    else:
        q_vals = q_table[state]
        max_val = max(q_vals)
        # Handle ties randomly
        best_actions = [i for i, v in enumerate(q_vals) if v == max_val]
        return random.choice(best_actions)

def draw_grid(drone_state, path_history=[]):
    """
    Draws the 2D grid in terminal using characters
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 45)
    print("      AERO-RL 2D DRONE NAVIGATOR")
    print("=" * 45)
    
    # Create path set for rendering
    path_set = set(path_history)
    
    for y in range(GRID_SIZE):
        row_str = " "
        for x in range(GRID_SIZE):
            cell = (x, y)
            if cell == drone_state:
                row_str += "🛸 " # Drone
            elif cell == GOAL_POS:
                row_str += "🏁 " # Target
            elif cell in OBSTACLES:
                row_str += "🧱 " # Obstacle Wall
            elif cell in path_set:
                row_str += "· "  # Path taken
            else:
                row_str += ". "  # Empty
        print(row_str)
    print("-" * 45)

def main():
    global EPSILON
    
    total_episodes = 60
    success_history = []
    
    print("Starting Training Loop. Drone will learn 2D navigation...")
    time.sleep(2)
    
    for ep in range(1, total_episodes + 1):
        state = START_POS
        done = False
        steps = 0
        ep_reward = 0.0
        path = [state]
        
        while not done:
            steps += 1
            action = select_action(state, EPSILON)
            next_state, reward, done, msg = step(state, action)
            
            # Bellman update
            old_q = q_table[state][action]
            max_next_q = max(q_table[next_state])
            new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
            q_table[state][action] = new_q
            
            # Move drone
            state = next_state
            path.append(state)
            ep_reward += reward
            
            # Animate only selected episodes so it trains faster
            # We animate the first 3 episodes and every 10th episode.
            if ep <= 3 or ep % 10 == 0:
                draw_grid(state, path)
                print(f" Episode: {ep} | Step: {steps} | Epsilon: {EPSILON:.2f}")
                print(f" Action: {ACTION_NAMES[action]}")
                print(f" Event: {msg} (Reward: {reward:.1f})")
                time.sleep(0.12)
            
            if done:
                is_success = (state == GOAL_POS)
                success_history.append(is_success)
                break
                
            if steps > 150:
                success_history.append(False)
                break
                
        # Epsilon decay
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            if EPSILON < MIN_EPSILON:
                EPSILON = MIN_EPSILON
                
        # Print summary for non-animated runs
        if not (ep <= 3 or ep % 10 == 0):
            recent_success = success_history[-10:]
            success_rate = (recent_success.count(True) / len(recent_success)) * 100 if recent_success else 0
            print(f"Episode {ep:2d}/{total_episodes:2d} finished | Steps: {steps:3d} | Reward: {ep_reward:6.1f} | Epsilon: {EPSILON:.2f} | Success Rate (last 10): {success_rate:.0f}%")
            
    # Print final policy map
    print("\nTraining completed! Final learned policy path:")
    state = START_POS
    eval_path = [state]
    done = False
    eval_steps = 0
    
    # Trace path with epsilon = 0 (pure exploitation)
    while not done and eval_steps < 30:
        eval_steps += 1
        action = select_action(state, 0.0) # Greedy action
        next_state, _, done, _ = step(state, action)
        if next_state == state: # stuck at boundary
            break
        state = next_state
        eval_path.append(state)
        if next_state in OBSTACLES:
            break
            
    draw_grid(GOAL_POS, eval_path)
    print("Optimal route plotted above as dots (·)!")
    if eval_path[-1] == GOAL_POS:
        print("🎉 Success! The drone successfully navigated the obstacle maze!")
    else:
        print("❌ The drone did not reach the target safely during the evaluation test.")
        
if __name__ == "__main__":
    main()
