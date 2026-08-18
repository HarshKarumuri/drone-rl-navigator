# Deep Reinforcement Learning for Autonomous Micro-UAV Obstacle Avoidance and Path Planning in Constrained Environments

**Author:** [Your Name]  
**Target Submission:** IEEE International Conference on Robotics and Automation (ICRA) / Portfolio Submission for ND Matrix

---

## Abstract
Autonomous navigation of Micro-Unmanned Aerial Vehicles (UAVs) in GPS-denied, obstacle-dense environments is a major challenge in robotics. Traditional path-planning algorithms (such as $A^*$ or Dijkstra) are computationally heavy and require complete structural maps of the environment beforehand. This paper proposes a model-free Reinforcement Learning (RL) approach for reactive path planning. We implement and compare a lookup table Q-learning agent in a discrete grid environment and a Deep Q-Network (DQN) agent operating over a continuous 6-dimensional telemetry vector. Our agents successfully converge to optimal, collision-free trajectories. We outline how this framework integrates with standard hardware (e.g., NVIDIA Jetson, PX4 autopilots) and simulation environments (AirSim/Gazebo), demonstrating a viable pipeline for real-world deployment.

---

## I. Introduction
Micro-UAVs are increasingly deployed for search-and-rescue, indoor inspection, and surveillance. Navigating these vehicles autonomously requires three core modules: localization, mapping, and path planning. Traditional path-planning systems map the 3D space first (e.g., using SLAM) and then compute a path. However, in dynamic environments where obstacles move or appear suddenly, pre-computed paths fail.

Reinforcement Learning (RL) offers a solution by combining perception and planning into a single policy loop. The drone learns directly through interaction: it takes actions, receives sensor readings, and modifies its behavioral policy based on numerical feedback.

---

## II. System Formulation

```
+--------------------------------------------------------------+
|                          STATE SPACE                         |
|  - UAV coordinates: (x, y)                                   |
|  - Lidar rangefinders: (d_up, d_down, d_left, d_right)       |
+------------------------------+-------------------------------+
                               |
                               v
+------------------------------+-------------------------------+
|                        POLICY NETWORK                        |
|  - Input layer: 6 dimensions                                 |
|  - Hidden layers: 2x 64-neuron Dense layers                  |
|  - Output layer: Q-values for actions                        |
+------------------------------+-------------------------------+
                               |
                               v
+------------------------------+-------------------------------+
|                         ACTION SPACE                         |
|  - Motor setpoints: [UP, DOWN, LEFT, RIGHT]                  |
+------------------------------+-------------------------------+
                               |
                               v
+------------------------------+-------------------------------+
|                       REWARD FUNCTION                        |
|  - Energy penalty: -1.0                                      |
|  - Shaping bonus: distance-to-goal reduction                 |
|  - Collision: -40.0 | Goal arrival: +100.0                   |
+--------------------------------------------------------------+
```

### A. State Space ($S$)
For the discrete simulator, the state is represented by coordinates:
$$S = \{ (x, y) \mid x, y \in [0, N-1] \}$$

For the continuous DQN agent, the state space is modeled as a 6-dimensional real-value vector representing the vehicle's spatial coordinates and relative proximity to walls:
$$S = \{ x, y, d_{up}, d_{down}, d_{left}, d_{right} \}$$
where $x, y$ are the drone's coordinates and $d_i$ are distance measurements from onboard rangefinder sensors.

### B. Action Space ($A$)
We define a discrete action space mapping to thrust vectors:
$$A = \{ 0, 1, 2, 3 \} \rightarrow \{ \text{Up}, \text{Down}, \text{Left}, \text{Right} \}$$

### C. Reward Function ($R$)
To ensure safety and speed, we implement a shaped reward function:

$$R(s, a, s') = 
\begin{cases} 
+100.0 & \text{if } s' \text{ is the Goal} \\
-40.0 & \text{if } s' \text{ is an Obstacle} \\
-1.0 + k \cdot (D(s, g) - D(s', g)) & \text{otherwise}
\end{cases}$$

Here, $D(s, g)$ is the distance from state $s$ to goal $g$, and $k$ is a scaling factor ($k=0.5$). The distance delta reward (shaping) guides the drone towards the goal in early learning phases when the goal has not yet been discovered.

---

## III. Algorithm Design

### A. Q-Learning (Discrete)
Q-learning maintains a lookup table containing expected cumulative rewards. The update rule is governed by the Bellman optimality equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### B. Deep Q-Networks (DQN) (Continuous)
For infinite states, we approximate Q-values using a neural network parameterised by weights $\theta$. We train this network by minimizing the Mean Squared Error (MSE) loss:

$$L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

To prevent oscillation, we use:
1.  **Experience Replay**: Storing experiences $(s, a, r, s', done)$ in a buffer and sampling randomly to break correlation.
2.  **Target Network ($\theta^-$)**: A copy of the network updated periodically to keep targets stable.

---

## IV. Hardware Deployment & Telemetry Bridge

```
 +------------------+     MAVLink (Commands)     +--------------------+
 |  Companion PC    | -------------------------> |  Flight Controller |
 |  (NVIDIA Jetson) |                            |  (Pixhawk / PX4)   |
 +------------------+                            +--------------------+
          ^                                                |
          | Sensor Data                                    | Motor Signals
          |                                                v
 +------------------+                            +--------------------+
 |   Depth Camera   |                            |     UAV Motors     |
 |   / LiDAR        |                            |    (Actuators)     |
 +------------------+                            +--------------------+
```

For real-world deployment (matching the systems engineered by **ND Matrix**):
1.  **Hardware Stack**: 
    *   **Flight Controller**: Pixhawk running PX4 autopilot software (manages low-level stability, roll/pitch/yaw).
    *   **Companion Computer**: NVIDIA Jetson Orin Nano (runs the PyTorch DQN model and processes LiDAR data).
2.  **Telemetry Protocol**: The companion computer communicates with the flight controller using **MAVLink** (Micro Air Vehicle Link) via **ROS 2** (Robot Operating System) and the `mavros` library.
3.  **Real-to-Sim Translation**: We first train the network in a high-fidelity simulator like **AirSim** or **Gazebo** to prevent real-world crashes. Once the policy stabilizes, the network weights are exported to a deployment script on the Jetson computer.

---

## V. Conclusion & Future Work
This work demonstrates that model-free reinforcement learning successfully trains micro-UAV agents to navigate complex 2D layouts. Future work will extend this framework to 3D state spaces containing altitude dimensions ($z$) and dynamic obstacles (like moving humans or other drones), utilizing PPO (Proximal Policy Optimization) algorithms.
