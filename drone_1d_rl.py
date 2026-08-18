# AERO-RL: 1D Drone Position Navigator
# Concept: A drone on a wire/line learning to navigate to a target landing pad.
# Target Audience: Explained for a 10-year-old, written with full depth.

import random
import time
import os
import sys

# Ensure UTF-8 output on Windows consoles to support drone emojis
sys.stdout.reconfigure(encoding='utf-8')

# --- STEP 1: DEFINE THE DRONE'S UNIVERSE (The Environment) ---
# Our drone can only move left or right along a straight line of spots.
# Think of the spots as a line of 11 floor tiles: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
GRID_SIZE = 11
START_POS = 2  # The drone starts at tile 2
GOAL_POS = 8   # The landing pad (target) is at tile 8

# Actions: 0 = Move Left, 1 = Move Right
ACTIONS = [0, 1]
ACTION_NAMES = {0: "LEFT (<-)", 1: "RIGHT (->)"}

# Hyperparameters: The knobs we turn to change how the drone learns
ALPHA = 0.20  # Learning Rate: How much the drone trusts new experiences (0 = learn nothing, 1 = forget the past instantly)
GAMMA = 0.90  # Discount Factor: How much the drone cares about future rewards (0 = short-sighted, 1 = long-term planner)
EPSILON = 0.5 # Exploration Rate: 50% chance of making a random move, 50% chance of doing what it thinks is best

# --- STEP 2: THE DRONE'S BRAIN (The Q-Table) ---
# A Q-table is a spreadsheet of memories.
# It has one row for each tile (0 to 10) and one column for each action (Left, Right).
# Q(state, action) is a score that tells the drone: "How good is it to take this action on this tile?"
# At the start, the drone knows nothing, so all scores are 0.0.
q_table = {}
for state in range(GRID_SIZE):
    q_table[state] = [0.0, 0.0]  # [Score for Left, Score for Right]

# --- STEP 3: THE PHYSICS & REWARDS (The Step Function) ---
def take_action(position, action):
    """
    Tells us what happens when the drone takes an action.
    Returns: (new_position, reward, is_goal_reached)
    """
    # Action: Left
    if action == 0:
        new_position = position - 1
    # Action: Right
    else:
        new_position = position + 1
        
    # Check if we hit the boundaries (falling off the wire)
    if new_position < 0 or new_position >= GRID_SIZE:
        # Penalty! And we don't move.
        return position, -10.0, False

    # Check if we reached the landing pad
    if new_position == GOAL_POS:
        # Huge reward! Mission complete!
        return new_position, 100.0, True

    # Every second of flying costs battery energy (step cost)
    # This pushes the drone to get to the goal as fast as possible.
    return new_position, -1.0, False

# --- STEP 4: VISUALIZING THE FLIGHT IN THE TERMINAL ---
def render(position, episode, step, last_action, reward):
    """
    Clears the screen and draws the line world as text.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print(f" AERO-RL 1D DRONE SIMULATOR (Episode {episode}, Step {step})")
    print("=" * 60)
    
    # Draw the line of tiles
    line = []
    for i in range(GRID_SIZE):
        if i == position:
            line.append("🛸") # The drone
        elif i == GOAL_POS:
            line.append("🏁") # The goal landing pad
        else:
            line.append(".")  # Empty tile
            
    print("  " + "  ".join(line))
    print("  " + "  ".join([f"{i:2d}" for i in range(GRID_SIZE)]))
    print("-" * 60)
    
    if last_action is not None:
        print(f" Last Action: {ACTION_NAMES[last_action]}")
        print(f" Last Reward: {reward}")
    else:
        print(" Starting new flight...")
        
    print("\n --- CURRENT Q-TABLE BRAIN ---")
    print(" Tile | Action LEFT (<-) | Action RIGHT (->)")
    print("------|------------------|-------------------")
    for state in range(GRID_SIZE):
        left_val = q_table[state][0]
        right_val = q_table[state][1]
        
        # Highlight current drone tile
        pointer = " <-- Drone" if state == position else ""
        print(f"  {state:2d}  |     {left_val:7.2f}      |      {right_val:7.2f}       {pointer}")
    print("=" * 60)

# --- STEP 5: THE TRAINING LOOP ---
def main():
    global EPSILON
    
    total_episodes = 10
    
    for episode in range(1, total_episodes + 1):
        position = START_POS
        step = 0
        done = False
        last_action = None
        reward = 0.0
        
        render(position, episode, step, last_action, reward)
        time.sleep(1.5)
        
        while not done:
            step += 1
            
            # 1. Epsilon-Greedy Choice: Explore vs. Exploit
            # Roll a random number between 0 and 1
            if random.random() < EPSILON:
                # Explore: Pick a random move
                action = random.choice(ACTIONS)
            else:
                # Exploit: Look at the Q-table and pick the action with the higher score
                left_score = q_table[position][0]
                right_score = q_table[position][1]
                if left_score > right_score:
                    action = 0
                elif right_score > left_score:
                    action = 1
                else:
                    action = random.choice(ACTIONS) # Tie-breaker
            
            # 2. Run the move in the physical world
            new_position, reward, done = take_action(position, action)
            
            # 3. Bellman Equation Brain Update
            # New Q = Old Q + Learning_Rate * [ Reward + Gamma * Best_Future_Q_Score - Old Q ]
            old_q = q_table[position][action]
            max_future_q = max(q_table[new_position])
            
            # This is the magic formula! It calculates how much we were wrong and adjusts our table.
            new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
            q_table[position][action] = new_q
            
            # Save action for render info
            last_action = action
            
            # Teleport drone to new spot
            position = new_position
            
            # Render the screen
            render(position, episode, step, last_action, reward)
            time.sleep(0.5) # slow down so we can see the flight
            
            if done:
                print(f"\n🎉 Woohoo! Drone reached the goal in {step} steps! Episode finished.")
                time.sleep(2.0)
                break
                
            if step >= 30:
                print("\n⚠️ Out of battery! Episode timed out.")
                time.sleep(2.0)
                break
        
        # Slowly decrease exploration rate. As the drone gets smarter, it should explore less.
        EPSILON *= 0.85
        
    print("\nTraining completed! Run the script again to see if it learns different choices!")

if __name__ == "__main__":
    main()
