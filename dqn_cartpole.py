import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# hyperparameters
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_FREQ = 1000
MAX_EPISODES = 3000
MAX_STEPS = 500
SOLVED_THRESHOLD = 195

# eps decay parameters
EPS_START = 1.0   # chooses random action 100% of the time
EPS_END = 0.01    # chooses best action 99% of the time
EPS_DECAY = 5_000   # larger: slower decay smaller: faster decay

# q-network definition
class QNetwork(nn.Module):
    # state -> hidden -> hidden -> action
    def __init__(self, obs_dim, n_actions):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.net(x)

# replay buffer definition
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        state = np.array(state, dtype=np.float32).reshape(-1)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# trains agent
def train_dqn():
    """
    train dqn agent on cartpole using epsilon decay
    returns:
       - policy_net: trained q-network
       - episode_rewards: list of total rewards per episode
    """
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]  # cartpole => 4
    n_actions = env.action_space.n            # cartpole => 2

    policy_net = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)

    # initialize target network weights to match policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    # track epsilon and total steps across training
    epsilon = EPS_START
    steps_done = 0

    # warm up replay buffer
    obs, info = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, info = env.reset()

    # main training loop
    episode_rewards = []
    for episode in range(MAX_EPISODES):
        obs, info = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            steps_done += 1

            # epsilon-greedy exploration
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = policy_net(obs_t)
                    action = q_values.argmax(dim=1).item()

            # take step
            next_obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            total_reward += reward

            # store transition in replay buffer
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # epsilon decay
            epsilon = max(EPS_END, EPS_START - steps_done / EPS_DECAY)

            # sample batch from replay buffer
            states, actions_b, rewards_b, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # convert to torch tensors
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions_b).unsqueeze(1)
            rewards_t = torch.FloatTensor(rewards_b)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones)

            # current q (for action taken)
            q_values_current = policy_net(states_t).gather(1, actions_t).squeeze(1)

            # target q (using the target network)
            with torch.no_grad():
                max_next_q_values = target_net(next_states_t).max(dim=1)[0]
                q_values_target = rewards_t + GAMMA * max_next_q_values * (1 - dones_t)

            # loss (mse current vs target)
            loss = nn.MSELoss()(q_values_current, q_values_target)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # periodically sync target network
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)

        # print average reward over last 10 episodes    
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

        # check if "solved"
        if len(episode_rewards) >= 100:
            recent_avg = np.mean(episode_rewards[-100:])
            if recent_avg >= SOLVED_THRESHOLD:
                print(f"Solved in episode {episode+1} with average reward {recent_avg:.2f}!")
                break

    env.close()
    return policy_net, episode_rewards

# visualizes and evaluates the trained agent
def evaluate_agent(policy_net, env_name=ENV_NAME, episodes=3):
    """
    Runs the trained policy_net in real-time rendering mode.
    """

    # create a new environment with a pop-up window
    env = gym.make(env_name, render_mode="human")
    policy_net.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            env.render()
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                q_values = policy_net(obs_t)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"[Evaluation] Episode {ep+1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    # train
    policy_net, episode_rewards = train_dqn()

    # evaluate
    evaluate_agent(policy_net, ENV_NAME, episodes=3)
