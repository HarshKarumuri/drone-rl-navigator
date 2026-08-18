// AERO-RL: Reinforcement Learning Drone Navigator
// Mathematical and visual rendering engine

class DroneQLearning {
    constructor() {
        this.gridSize = 10;
        this.resetMap();
        
        // Learning parameters (hyperparameters)
        this.epsilon = 1.0;  // Exploration rate
        this.alpha = 0.15;   // Learning rate
        this.gamma = 0.90;   // Discount factor
        this.epsilonDecay = 0.995;
        this.minEpsilon = 0.05;

        // Statistics
        this.episodeCount = 0;
        this.stepsCount = 0;
        this.cumulativeReward = 0;
        this.lastReward = 0;
        this.historyRewards = [];
        this.successCount = 0;
        this.runHistory = []; // Tracks if episodes ended in Goal (true) or Crash (false)

        // Animation and control variables
        this.isPlaying = false;
        this.speedFps = 15;
        this.animationFrameId = null;
        this.lastFrameTime = 0;
        this.stepMode = false;
        
        // View settings
        this.showHeatmap = true;
        this.showArrows = true;

        // Rotor animation angle
        this.rotorAngle = 0;

        // Initialize UI bindings
        this.initUI();
        
        // Initial draw
        this.draw();
    }

    resetMap() {
        // Start State (Top-Left)
        this.startState = { x: 0, y: 0 };
        // Goal State (Bottom-Right)
        this.goalState = { x: this.gridSize - 1, y: this.gridSize - 1 };
        
        // Drone Position
        this.drone = { ...this.startState };
        
        // Map Grid: 0 = Empty, 1 = Obstacle
        this.grid = Array(this.gridSize).fill(null).map(() => Array(this.gridSize).fill(0));
        
        // Set static obstacles (forming a challenging maze)
        const obstacles = [
            {x: 2, y: 0}, {x: 2, y: 1}, {x: 2, y: 2},
            {x: 4, y: 4}, {x: 4, y: 5}, {x: 4, y: 6}, {x: 4, y: 7},
            {x: 7, y: 2}, {x: 7, y: 3}, {x: 7, y: 4},
            {x: 1, y: 6}, {x: 2, y: 6}, {x: 3, y: 6},
            {x: 7, y: 7}, {x: 8, y: 7}, {x: 6, y: 8}
        ];
        
        obstacles.forEach(obs => {
            if (!(obs.x === this.startState.x && obs.y === this.startState.y) &&
                !(obs.x === this.goalState.x && obs.y === this.goalState.y)) {
                this.grid[obs.y][obs.x] = 1;
            }
        });

        // Initialize Q-table: State key "x,y" -> array of size 4 [Up, Down, Left, Right]
        // 0: Up, 1: Down, 2: Left, 3: Right
        this.qTable = {};
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                this.qTable[`${x},${y}`] = [0, 0, 0, 0];
            }
        }
        
        this.drone = { ...this.startState };
        this.stepsCount = 0;
        this.cumulativeReward = 0;
        this.pathHistory = [{ ...this.drone }];
    }

    initUI() {
        this.canvas = document.getElementById('arena-canvas');
        this.ctx = this.canvas.getContext('2d');

        // Buttons
        this.btnStep = document.getElementById('btn-step');
        this.btnTrainEpisode = document.getElementById('btn-train-episode');
        this.btnAutoTrain = document.getElementById('btn-auto-train');
        this.btnReset = document.getElementById('btn-reset');

        // Sliders
        this.sliderEpsilon = document.getElementById('slider-epsilon');
        this.sliderAlpha = document.getElementById('slider-alpha');
        this.sliderGamma = document.getElementById('slider-gamma');
        this.sliderSpeed = document.getElementById('slider-speed');

        // Displays
        this.valEpsilon = document.getElementById('val-epsilon');
        this.valAlpha = document.getElementById('val-alpha');
        this.valGamma = document.getElementById('val-gamma');
        this.valSpeed = document.getElementById('val-speed');

        // Telemetry
        this.telPos = document.getElementById('tel-pos');
        this.telEpisode = document.getElementById('tel-episode');
        this.telSteps = document.getElementById('tel-steps');
        this.telReward = document.getElementById('tel-reward');
        this.telSuccessRate = document.getElementById('tel-success-rate');
        this.telTotalReward = document.getElementById('tel-total-reward');
        this.explainerLog = document.getElementById('explainer-log');

        // Toggles
        this.toggleHeatmap = document.getElementById('toggle-heatmap');
        this.toggleArrows = document.getElementById('toggle-arrows');

        // Event listeners
        this.btnStep.addEventListener('click', () => this.triggerStep());
        this.btnTrainEpisode.addEventListener('click', () => this.trainOneEpisode());
        this.btnAutoTrain.addEventListener('click', () => this.toggleAutoTrain());
        this.btnReset.addEventListener('click', () => {
            this.resetMap();
            this.episodeCount = 0;
            this.successCount = 0;
            this.runHistory = [];
            this.epsilon = 1.0;
            this.sliderEpsilon.value = 1.0;
            this.valEpsilon.textContent = '1.00';
            this.updateTelemetry();
            this.logMessage("Map regenerated. Q-Table cleared. Epsilon reset to 1.0 (pure exploration). Let's learn from scratch!");
            this.draw();
        });

        this.sliderEpsilon.addEventListener('input', (e) => {
            this.epsilon = parseFloat(e.target.value);
            this.valEpsilon.textContent = this.epsilon.toFixed(2);
        });

        this.sliderAlpha.addEventListener('input', (e) => {
            this.alpha = parseFloat(e.target.value);
            this.valAlpha.textContent = this.alpha.toFixed(2);
        });

        this.sliderGamma.addEventListener('input', (e) => {
            this.gamma = parseFloat(e.target.value);
            this.valGamma.textContent = this.gamma.toFixed(2);
        });

        this.sliderSpeed.addEventListener('input', (e) => {
            this.speedFps = parseInt(e.target.value);
            this.valSpeed.textContent = this.speedFps;
        });

        this.toggleHeatmap.addEventListener('change', (e) => {
            this.showHeatmap = e.target.checked;
            this.draw();
        });

        this.toggleArrows.addEventListener('change', (e) => {
            this.showArrows = e.target.checked;
            this.draw();
        });
    }

    logMessage(text) {
        this.explainerLog.innerHTML = text;
        this.explainerLog.scrollTop = 0; // Keep the latest viewable
    }

    // RL MATH: Epsilon-Greedy Action Selection
    selectAction(stateKey) {
        if (Math.random() < this.epsilon) {
            // EXPLORE: Pick random action
            return Math.floor(Math.random() * 4);
        } else {
            // EXPLOIT: Pick best action based on current Q-table
            const qValues = this.qTable[stateKey];
            let maxVal = -Infinity;
            let bestActions = [];
            for (let i = 0; i < 4; i++) {
                if (qValues[i] > maxVal) {
                    maxVal = qValues[i];
                    bestActions = [i];
                } else if (qValues[i] === maxVal) {
                    bestActions.push(i);
                }
            }
            // Tie-breaker
            return bestActions[Math.floor(Math.random() * bestActions.length)];
        }
    }

    // RL ENVIRONMENT: Apply action, return new state and reward
    step(action) {
        let nextX = this.drone.x;
        let nextY = this.drone.y;

        // Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if (action === 0) nextY--;
        else if (action === 1) nextY++;
        else if (action === 2) nextX--;
        else if (action === 3) nextX++;

        // Check boundary limits
        if (nextX < 0 || nextX >= this.gridSize || nextY < 0 || nextY >= this.gridSize) {
            // Out of bounds penalty, stay in place
            return {
                state: { ...this.drone },
                reward: -10,
                status: 'boundary',
                msg: `Ouch! I tried to fly out of bounds at (${nextX}, ${nextY}). My distance sensors stopped me! Penalty: -10.`
            };
        }

        // Check obstacles
        if (this.grid[nextY][nextX] === 1) {
            // Crash! Reset to start, high penalty
            return {
                state: { x: nextX, y: nextY }, // returned for updating Q-value
                reward: -40,
                status: 'crash',
                msg: `BOOM! 💥 I crashed into an obstacle shield at (${nextX}, ${nextY})! Instant penalty: -40. Rebooting telemetry at start position.`
            };
        }

        // Check goal
        if (nextX === this.goalState.x && nextY === this.goalState.y) {
            return {
                state: { x: nextX, y: nextY },
                reward: 100,
                status: 'goal',
                msg: `SUCCESS! 🎉 Safe arrival at Target Goal (${nextX}, ${nextY})! Energy reward: +100.`
            };
        }

        // Regular step (UAV flying energy cost)
        // Give a tiny distance shaping reward: moving closer to goal gets a slightly smaller penalty
        const distBefore = Math.abs(this.drone.x - this.goalState.x) + Math.abs(this.drone.y - this.goalState.y);
        const distAfter = Math.abs(nextX - this.goalState.x) + Math.abs(nextY - this.goalState.y);
        const distanceBonus = distAfter < distBefore ? 0.2 : -0.2;
        const stepReward = -1 + distanceBonus;

        return {
            state: { x: nextX, y: nextY },
            reward: stepReward,
            status: 'fly',
            msg: `Cruising safe. Moved to (${nextX}, ${nextY}). Flight energy cost: ${stepReward.toFixed(1)}.`
        };
    }

    // Core Q-Learning loop step
    learnStep() {
        const currentState = { ...this.drone };
        const stateKey = `${currentState.x},${currentState.y}`;
        
        // 1. Choose action
        const action = this.selectAction(stateKey);
        
        // 2. Perform action & observe environment
        const outcome = this.step(action);
        const nextState = outcome.state;
        const reward = outcome.reward;
        
        // 3. Update Q-Table using Bellman Equation
        const nextStateKey = `${nextState.x},${nextState.y}`;
        const maxFutureQ = Math.max(...this.qTable[nextStateKey]);
        const oldQ = this.qTable[stateKey][action];
        
        // Bellman Formula: Q(s,a) = Q(s,a) + alpha * [ reward + gamma * max(Q(s',a')) - Q(s,a) ]
        const newQ = oldQ + this.alpha * (reward + this.gamma * maxFutureQ - oldQ);
        this.qTable[stateKey][action] = newQ;

        // Apply state transition
        this.lastReward = reward;
        this.cumulativeReward += reward;
        this.stepsCount++;

        const actionNames = ['UP (↑)', 'DOWN (↓)', 'LEFT (←)', 'RIGHT (→)'];
        let logText = `<b>State:</b> (${currentState.x}, ${currentState.y}) → chosen action: <b>${actionNames[action]}</b>.<br>`;
        logText += `${outcome.msg}<br>`;
        logText += `<b>Math Update:</b> Q-value changed from ${oldQ.toFixed(2)} to <b>${newQ.toFixed(2)}</b> using reward ${reward.toFixed(1)}.`;

        if (outcome.status === 'crash') {
            // Update stats
            this.runHistory.push(false);
            if (this.runHistory.length > 50) this.runHistory.shift();
            
            // Move drone to start
            this.drone = { ...this.startState };
            this.pathHistory = [{ ...this.drone }];
            this.episodeCount++;
            
            // Decay epsilon
            this.decayEpsilon();
            this.updateTelemetry();
            this.logMessage(logText + `<br><br><span style="color: var(--neon-red)">Episode ${this.episodeCount} ended in a crash!</span>`);
            return 'episode_end';
        } else if (outcome.status === 'goal') {
            // Update stats
            this.runHistory.push(true);
            if (this.runHistory.length > 50) this.runHistory.shift();
            this.successCount++;
            
            // Add drone path
            this.pathHistory.push({ ...nextState });
            this.drone = { ...nextState };
            this.draw();
            
            // Wait brief moment and reset drone to start
            setTimeout(() => {
                this.drone = { ...this.startState };
                this.pathHistory = [{ ...this.drone }];
                this.episodeCount++;
                this.decayEpsilon();
                this.updateTelemetry();
            }, 100);
            
            this.logMessage(logText + `<br><br><span style="color: var(--neon-green)">Episode ${this.episodeCount} arrived successfully!</span>`);
            return 'episode_end';
        } else {
            // Regular step transition
            this.drone = { ...nextState };
            this.pathHistory.push({ ...this.drone });
            this.updateTelemetry();
            this.logMessage(logText);
            return 'step';
        }
    }

    decayEpsilon() {
        if (this.epsilon > this.minEpsilon) {
            this.epsilon *= this.epsilonDecay;
            if (this.epsilon < this.minEpsilon) this.epsilon = this.minEpsilon;
            this.sliderEpsilon.value = this.epsilon;
            this.valEpsilon.textContent = this.epsilon.toFixed(2);
        }
    }

    triggerStep() {
        this.isPlaying = false;
        this.btnAutoTrain.textContent = '🔥 Auto-Train';
        this.btnAutoTrain.classList.remove('btn-danger');
        this.btnAutoTrain.classList.add('btn-primary');
        
        this.learnStep();
        this.draw();
    }

    trainOneEpisode() {
        this.isPlaying = false;
        this.btnAutoTrain.textContent = '🔥 Auto-Train';
        this.btnAutoTrain.classList.remove('btn-danger');
        this.btnAutoTrain.classList.add('btn-primary');

        let steps = 0;
        const maxSteps = 200;
        
        const runEpisode = () => {
            const result = this.learnStep();
            this.draw();
            steps++;
            if (result !== 'episode_end' && steps < maxSteps) {
                setTimeout(runEpisode, 1000 / this.speedFps);
            }
        };
        runEpisode();
    }

    toggleAutoTrain() {
        if (this.isPlaying) {
            // Stop
            this.isPlaying = false;
            this.btnAutoTrain.textContent = '🔥 Auto-Train';
            this.btnAutoTrain.classList.remove('btn-danger');
            this.btnAutoTrain.classList.add('btn-primary');
        } else {
            // Start
            this.isPlaying = true;
            this.btnAutoTrain.textContent = '⏸ Pause Training';
            this.btnAutoTrain.classList.remove('btn-primary');
            this.btnAutoTrain.classList.add('btn-danger');
            this.lastFrameTime = performance.now();
            this.runAutoTrainLoop();
        }
    }

    runAutoTrainLoop() {
        if (!this.isPlaying) return;

        const now = performance.now();
        const interval = 1000 / this.speedFps;
        const elapsed = now - this.lastFrameTime;

        if (elapsed >= interval) {
            this.learnStep();
            this.draw();
            this.lastFrameTime = now - (elapsed % interval);
        }

        requestAnimationFrame(() => this.runAutoTrainLoop());
    }

    updateTelemetry() {
        this.telPos.textContent = `(${this.drone.x}, ${this.drone.y})`;
        this.telEpisode.textContent = this.episodeCount;
        this.telSteps.textContent = this.stepsCount;
        
        this.telReward.textContent = this.lastReward.toFixed(1);
        if (this.lastReward > 0) {
            this.telReward.className = 'tel-val positive';
        } else {
            this.telReward.className = 'tel-val negative';
        }

        // Success rate in the last 50 runs
        const successRate = this.runHistory.length > 0 
            ? Math.round((this.runHistory.filter(x => x).length / this.runHistory.length) * 100)
            : 0;
        
        this.telSuccessRate.textContent = `${successRate}%`;
        this.telTotalReward.textContent = Math.round(this.cumulativeReward);
    }

    // DRAWING STUFF ON CANVAS
    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const cellSize = width / this.gridSize;

        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);

        // Draw grid squares and Heatmap
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                const stateKey = `${x},${y}`;
                const isObstacle = this.grid[y][x] === 1;
                const isGoal = x === this.goalState.x && y === this.goalState.y;

                if (isObstacle) {
                    // Draw obstacle with tech hazard lines
                    this.ctx.fillStyle = '#1e1c2a';
                    this.ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                    
                    this.ctx.strokeStyle = 'rgba(255, 153, 0, 0.4)';
                    this.ctx.lineWidth = 2;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x * cellSize, y * cellSize);
                    this.ctx.lineTo((x + 1) * cellSize, (y + 1) * cellSize);
                    this.ctx.moveTo((x + 1) * cellSize, y * cellSize);
                    this.ctx.lineTo(x * cellSize, (y + 1) * cellSize);
                    this.ctx.stroke();
                } else if (isGoal) {
                    // Draw target goal with glowing green tint
                    this.ctx.fillStyle = 'rgba(0, 255, 136, 0.15)';
                    this.ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                } else if (this.showHeatmap) {
                    // Heatmap based on V(s) = max(Q(s, a))
                    const qValues = this.qTable[stateKey];
                    const maxQ = Math.max(...qValues);
                    
                    if (maxQ > 0) {
                        // Max potential positive reward is +100
                        const intensity = Math.min(maxQ / 80, 0.85); // cap opacity
                        this.ctx.fillStyle = `rgba(0, 225, 255, ${intensity})`;
                        this.ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                    } else if (maxQ < 0) {
                        // Negative values glow red
                        const intensity = Math.min(Math.abs(maxQ) / 40, 0.5);
                        this.ctx.fillStyle = `rgba(255, 51, 102, ${intensity})`;
                        this.ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                    }
                }

                // Grid cell borders
                this.ctx.strokeStyle = 'rgba(0, 225, 255, 0.08)';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize);

                // Draw Policy Arrows
                if (this.showArrows && !isObstacle && !isGoal) {
                    const qValues = this.qTable[stateKey];
                    const maxQ = Math.max(...qValues);
                    
                    // Only draw if there's any knowledge learned
                    if (Math.abs(maxQ) > 0.0001) {
                        const bestAction = qValues.indexOf(maxQ);
                        this.drawArrow(x, y, bestAction, cellSize);
                    }
                }
            }
        }

        // Draw path history line
        if (this.pathHistory.length > 1) {
            this.ctx.beginPath();
            this.ctx.strokeStyle = '#ffd700'; // Gold color
            this.ctx.lineWidth = 3;
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
            
            // Offset path slightly to center of cell
            const offset = cellSize / 2;
            this.ctx.moveTo(this.pathHistory[0].x * cellSize + offset, this.pathHistory[0].y * cellSize + offset);
            for (let i = 1; i < this.pathHistory.length; i++) {
                this.ctx.lineTo(this.pathHistory[i].x * cellSize + offset, this.pathHistory[i].y * cellSize + offset);
            }
            this.ctx.stroke();
        }

        // Draw Goal Target Icon
        const goalOffset = cellSize / 2;
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.arc(this.goalState.x * cellSize + goalOffset, this.goalState.y * cellSize + goalOffset, cellSize * 0.35, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
        this.ctx.fill();
        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 2.5;
        this.ctx.stroke();
        
        // Draw inner target dot
        this.ctx.beginPath();
        this.ctx.arc(this.goalState.x * cellSize + goalOffset, this.goalState.y * cellSize + goalOffset, cellSize * 0.1, 0, Math.PI * 2);
        this.ctx.fillStyle = '#00ff88';
        this.ctx.fill();
        
        // Write letter "G"
        this.ctx.fillStyle = '#fff';
        this.ctx.font = `bold ${cellSize * 0.3}px var(--font-header)`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText("G", this.goalState.x * cellSize + goalOffset, this.goalState.y * cellSize + goalOffset - 1);
        this.ctx.restore();

        // Draw Drone (✈)
        this.drawDrone(cellSize);
    }

    drawArrow(x, y, action, cellSize) {
        const cx = x * cellSize + cellSize / 2;
        const cy = y * cellSize + cellSize / 2;
        const length = cellSize * 0.25;
        
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        this.ctx.beginPath();
        
        // Draw tiny arrow based on direction
        // 0=Up, 1=Down, 2=Left, 3=Right
        if (action === 0) { // Up
            this.ctx.moveTo(cx, cy + length / 2);
            this.ctx.lineTo(cx, cy - length / 2);
            this.ctx.lineTo(cx - 4, cy - length / 2 + 4);
            this.ctx.moveTo(cx, cy - length / 2);
            this.ctx.lineTo(cx + 4, cy - length / 2 + 4);
        } else if (action === 1) { // Down
            this.ctx.moveTo(cx, cy - length / 2);
            this.ctx.lineTo(cx, cy + length / 2);
            this.ctx.lineTo(cx - 4, cy + length / 2 - 4);
            this.ctx.moveTo(cx, cy + length / 2);
            this.ctx.lineTo(cx + 4, cy + length / 2 - 4);
        } else if (action === 2) { // Left
            this.ctx.moveTo(cx + length / 2, cy);
            this.ctx.lineTo(cx - length / 2, cy);
            this.ctx.lineTo(cx - length / 2 + 4, cy - 4);
            this.ctx.moveTo(cx - length / 2, cy);
            this.ctx.lineTo(cx - length / 2 + 4, cy + 4);
        } else if (action === 3) { // Right
            this.ctx.moveTo(cx - length / 2, cy);
            this.ctx.lineTo(cx + length / 2, cy);
            this.ctx.lineTo(cx + length / 2 - 4, cy - 4);
            this.ctx.moveTo(cx + length / 2, cy);
            this.ctx.lineTo(cx + length / 2 - 4, cy + 4);
        }
        
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawDrone(cellSize) {
        const cx = this.drone.x * cellSize + cellSize / 2;
        const cy = this.drone.y * cellSize + cellSize / 2;
        const r = cellSize * 0.35; // outer diameter

        this.ctx.save();
        
        // Drone Glow Effect
        const grad = this.ctx.createRadialGradient(cx, cy, 2, cx, cy, r);
        grad.addColorStop(0, 'rgba(0, 225, 255, 0.4)');
        grad.addColorStop(1, 'rgba(0, 225, 255, 0.0)');
        this.ctx.fillStyle = grad;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, r * 1.5, 0, Math.PI * 2);
        this.ctx.fill();

        // Cross-arm drone fuselage
        this.ctx.strokeStyle = '#00e1ff';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        // Arm 1: top-left to bottom-right
        this.ctx.moveTo(cx - r * 0.7, cy - r * 0.7);
        this.ctx.lineTo(cx + r * 0.7, cy + r * 0.7);
        // Arm 2: top-right to bottom-left
        this.ctx.moveTo(cx + r * 0.7, cy - r * 0.7);
        this.ctx.lineTo(cx - r * 0.7, cy + r * 0.7);
        this.ctx.stroke();

        // Central Pod (hull)
        this.ctx.fillStyle = '#070a13';
        this.ctx.strokeStyle = '#00e1ff';
        this.ctx.lineWidth = 2.5;
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, r * 0.35, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.stroke();

        // Telemetry target lens dot
        this.ctx.fillStyle = '#00ff88';
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        this.ctx.fill();

        // Spinning rotors on arms!
        this.rotorAngle += 0.25; // Spin increment
        const rotorR = r * 0.3; // Rotor blade radius
        const armOffsets = [
            [-r * 0.7, -r * 0.7], // top-left
            [r * 0.7, -r * 0.7],  // top-right
            [-r * 0.7,  r * 0.7], // bottom-left
            [r * 0.7,   r * 0.7]  // bottom-right
        ];

        this.ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        this.ctx.lineWidth = 1.5;
        armOffsets.forEach(([ox, oy]) => {
            const rx = cx + ox;
            const ry = cy + oy;

            // Draw rotor motor hub
            this.ctx.fillStyle = '#00e1ff';
            this.ctx.beginPath();
            this.ctx.arc(rx, ry, 2.5, 0, Math.PI * 2);
            this.ctx.fill();

            // Draw spinning rotor blades
            this.ctx.beginPath();
            this.ctx.moveTo(rx - Math.cos(this.rotorAngle) * rotorR, ry - Math.sin(this.rotorAngle) * rotorR);
            this.ctx.lineTo(rx + Math.cos(this.rotorAngle) * rotorR, ry + Math.sin(this.rotorAngle) * rotorR);
            this.ctx.stroke();
        });

        this.ctx.restore();
    }
}

// Instantiate dashboard when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    window.simulator = new DroneQLearning();
});
