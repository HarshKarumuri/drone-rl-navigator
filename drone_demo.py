# AERO-RL: Step-by-Step 1D Drone Learning Demonstration
# Prints step logs line-by-line (no screen clearing) to display directly in the chat.

import random
import time

GRID_SIZE = 11
START_POS = 2
GOAL_POS = 8

# Q-table: tile position -> [Left score, Right score]
q_table = {i: [0.0, 0.0] for i in range(GRID_SIZE)}

# Learning parameters
ALPHA = 0.20
GAMMA = 0.90
EPSILON = 0.6  # High exploration to show random decisions

def get_action(pos):
    # Epsilon-greedy
    if random.random() < EPSILON:
        return random.choice([0, 1])
    else:
        return 0 if q_table[pos][0] > q_table[pos][1] else 1

def run_demo():
    print("=" * 65)
    print("      AERO-RL 1D DRONE NAVIGATION LIVE CHAT DEMONSTRATION")
    print("=" * 65)
    print(f"Starting Drone at position {START_POS}. Target landing pad is at {GOAL_POS}.\n")

    pos = START_POS
    step = 0
    done = False
    
    while not done and step < 25:
        step += 1
        
        # 1. Action selection
        action = get_action(pos)
        action_name = "LEFT (<-)" if action == 0 else "RIGHT (->)"
        
        # 2. Apply action
        new_pos = pos
        if action == 0:
            new_pos -= 1
        else:
            new_pos += 1
            
        # Check boundary
        if new_pos < 0 or new_pos >= GRID_SIZE:
            reward = -10.0
            new_pos = pos
            outcome_msg = "Hit boundary shield! Returned back."
        elif new_pos == GOAL_POS:
            reward = 100.0
            done = True
            outcome_msg = "SUCCESS! Reached landing pad."
        else:
            reward = -1.0
            outcome_msg = "Safe cruising step."
            
        # 3. Q-value update
        old_q = q_table[pos][action]
        max_future_q = max(q_table[new_pos])
        new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
        q_table[pos][action] = new_q
        
        # 4. Render ASCII map
        line = []
        for i in range(GRID_SIZE):
            if i == pos:
                line.append("D")
            elif i == GOAL_POS:
                line.append("G")
            else:
                line.append(".")
        map_str = " ".join(line)
        
        # Print logs
        print(f"--- STEP {step} ---")
        print(f"State map:  {map_str}")
        print(f"Action:     {action_name}")
        print(f"Outcome:    {outcome_msg} (Reward: {reward})")
        print(f"Q-Update:   Q(state={pos}, action={action_name}) updated from {old_q:.2f} to {new_q:.2f}")
        print(f"Q-Brain:    Position {pos} scores -> Left: {q_table[pos][0]:.2f} | Right: {q_table[pos][1]:.2f}")
        print("-" * 65)
        
        # Move drone
        pos = new_pos
        time.sleep(0.1) # Brief pause
        
    if done:
        print("\n[SUCCESS] Episode successfully finished! Drone reached the landing pad.")
    else:
        print("\n[TIMEOUT] Episode timed out.")

if __name__ == "__main__":
    run_demo()
