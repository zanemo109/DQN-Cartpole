# CartPole DQN Project

## Overview
This repository contains a **Deep Q-Network (DQN)** implementation for solving the classic **CartPole-v1** environment using **PyTorch**. It demonstrates:

- Reinforcement Learning with Deep Q-Networks
- Epsilon-greedy exploration strategy
- Experience replay buffer
- Target network for stable learning

## Key Components
- [dqn_cartpole.py](cci:7://file:///Users/zanemogannam/Desktop/projects/DQN/dqn_cartpole.py:0:0-0:0): Main implementation of the DQN algorithm
- Replay buffer for experience storage
- Neural network for Q-value approximation
- Training and evaluation functions

## Dependencies
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib

## Algorithm Overview
The Deep Q-Network (DQN) learns to balance a pole on a moving cart by:
1. Observing the environment state
2. Selecting actions using an epsilon-greedy policy
3. Storing experiences in a replay buffer
4. Training a neural network to predict Q-values
5. Updating the policy network through backpropagation

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python dqn_cartpole.py`

## Results
The agent learns to balance the pole for extended periods, demonstrating the effectiveness of the DQN algorithm.

## References
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- OpenAI Gym CartPole-v1 environment