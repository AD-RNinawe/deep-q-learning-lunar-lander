# Deep Q-Learning Agent for Lunar Lander (PyTorch)

This project implements a Deep Q-Network (DQN) agent using PyTorch to solve the LunarLander-v2 environment from OpenAI Gymnasium. The agent learns to land a spacecraft safely by interacting with the environment and optimizing a neural network that approximates the optimal Q-function.

---

## Problem Statement

The Lunar Lander problem is a classic reinforcement learning task where an agent must learn a control policy to land a spacecraft safely between designated markers.

The environment provides:

- State space: 8 continuous variables (position, velocity, angle, angular velocity, leg contact)
- Action space: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)

The objective is to maximize cumulative reward through optimal decision-making.

---

## Methodology

This project implements a Deep Q-Learning algorithm with the following components:

- Neural network for Q-value approximation
- Experience Replay Buffer for stable learning
- Epsilon-greedy exploration strategy
- Mini-batch gradient descent using PyTorch

Unlike the full DQN algorithm, this implementation does not use a target network. This simplified setup was used to study learning behavior and convergence properties.

---

## Neural Network Architecture

The Q-network is a fully connected feedforward neural network:

Input Layer: 8 neurons (state space)  
Hidden Layer 1: Fully connected + ReLU  
Hidden Layer 2: Fully connected + ReLU  
Output Layer: 4 neurons (Q-value for each action)

The network learns to approximate:

Q(s, a)

which represents the expected future reward of taking action a in state s.

---

## Training Process

Training loop:

1. Agent interacts with environment
2. Stores experience in replay buffer
3. Samples mini-batches from replay buffer
4. Updates neural network using Bellman equation
5. Gradually reduces epsilon to shift from exploration to exploitation

---

## Training Results

Training performance is tracked using average episode rewards.

The training scores over 800 episodes has been recorded in the training_scores.xlsx

The increasing reward trend indicates that the agent successfully learns improved landing strategies over time.

---

## Demonstration

A short demonstration of the trained agent is available:

results/lunar_lander.mp4

This shows the agent interacting with the environment after training.

![Lunar Lander Agent](assets/lunar_lander_demo.gif)

---

## Technologies Used

- Python
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib
- Pandas

---

## Repository Structure

deep-q-learning-lunar-lander/
│
├── README.md
├── Deep_Q_Learning_Lunar_Lander.ipynb
├── requirements.txt
│
├── results/
│ └── lunar_lander.mp4
│ └── training_scores.xlsx
│
├── assetss/
│ └── lunar_lander_demo.gif


---

## How to Run

Install dependencies:

pip install -r requirements.txt


Open and run the notebook:

Deep_Q_Learning_Lunar_Lander.ipynb


---

## Future Improvements

- Add target network for improved training stability
- Implement Double DQN
- Implement Dueling DQN architecture
- Add model checkpoint saving/loading
- Train for longer duration for higher performance

---

## Key Concepts Demonstrated

- Reinforcement Learning
- Deep Q-Learning
- Neural Networks using PyTorch
- Experience Replay
- Exploration vs Exploitation tradeoff
- Function approximation in RL

---



