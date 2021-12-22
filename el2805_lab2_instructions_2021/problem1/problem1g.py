import numpy as np

from DQN_network import MyNetwork
import gym
from DQN_agent import RandomAgent
import torch
import math
import matplotlib.pyplot as plt


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
N_episodes = 50                             # Number of episodes

#random agent
# Random agent initialization
agent = RandomAgent(n_actions)
episode_total_random = []       # this list contains the total reward per episode
#episode_number_of_steps = []   # this list contains the number of steps per episode
for i in range(N_episodes):
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0

    while not done:
        # Take a random action
        action = agent.forward(state)
        next_state, reward, done, _ = env.step(action)


        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1
    # Append episode reward and total number of steps
    episode_total_random.append(total_episode_reward)
    #episode_number_of_steps.append(t)


#my agent
nn = torch.load('neural-network-1.pth')
episode_total_my_agent = []  # this list contains the total reward per episode
#episode_number_of_steps = []  # this list contains the number of steps per episode
for i in range(N_episodes):
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Take a action according to nn
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32)
        x = nn(state_tensor)
        action = x.max(1)[1].item()
        next_state, reward, done, _ = env.step(action)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1
    # Append episode reward and total number of steps
    episode_total_my_agent.append(total_episode_reward)
    #episode_number_of_steps.append(t)


env.close()


plt.figure()
plt.plot(np.linspace(1,50,50),episode_total_random, label='Episode reward random')
plt.plot(np.linspace(1,50,50),episode_total_my_agent, label='Episode reward my agent')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.show()