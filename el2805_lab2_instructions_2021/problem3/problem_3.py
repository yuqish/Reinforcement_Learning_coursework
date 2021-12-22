# Yuqi Shao 199507014208

# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent
from collections import deque, namedtuple
import torch.optim as optim
from PPO_network import ActorNetwork
from PPO_network import CriticNetwork
import copy
import scipy.stats
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def index(self, a):
        return self.buffer.index(a)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def to_list(self):
        indices = range(len(self.buffer))

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]
        # batch.append(new_exp)

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


def compute_pi_probability(mu, sigma2, action):
    return torch.pow(2 * np.pi * sigma2, -1 / 2) * torch.exp(
    - torch.pow(action - mu, 2) / (2 * sigma2))

def train_actor_network(states_tensor, actions_tensor, phi_tensor, pi_old_tensor):
    t = len(actions)
    optimizer_actor.zero_grad()
    mu_new, sigma2_new = actor_network(states_tensor)
    pi_tensor = (compute_pi_probability(mu_new[:, 0], sigma2_new[:, 0], actions_tensor[:, 0]) *
     compute_pi_probability(mu_new[:, 1], sigma2_new[:, 1], actions_tensor[:, 1])).reshape(-1, )
    ratio = pi_tensor/pi_old_tensor
    c1 = torch.clamp(ratio, max=1+eps)
    c2 = torch.clamp(c1, min=1-eps)
    func = torch.clamp(phi_tensor*ratio, max=phi_tensor*c2)
    loss = sum(func)*(-1/t)
    loss.backward()
    # Clip gradient norm to 1
    #nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=1)
    optimizer_actor.step()


def train_critic_network(G_tensor, states_tensor):
    optimizer_critic.zero_grad()
    values = critic_network(states_tensor)
    values = torch.reshape(values, (-1,))
    # Compute gradient
    loss = nn.functional.mse_loss(values, G_tensor)
    loss.backward()
    # Clip gradient norm to 1
    #nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=1)
    optimizer_critic.step()


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# initialize buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# Parameters
N_episodes = 1600               # Number of episodes to run for training
discount_factor = 0.99         # Value of gamma
#discount_factor = 0.5         # Value of gamma
n_ep_running_average = 50      # Running average of 20 episodes
m = len(env.action_space.high) # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality
M = 10  # training epochs
LR_actor = 1e-5
LR_critic = 1e-3
L = 100000   # buffer size is large enough for storing any trajectories
eps = 0.2   # bound on policy
#eps = 0.3   # bound on policy

### Create networks ###
actor_network = ActorNetwork(state_size=dim_state, action_size=m)
tar_actor_network = copy.deepcopy(actor_network)
critic_network = CriticNetwork(state_size=dim_state)
tar_critic_network = copy.deepcopy(critic_network)

### Create optimizer ###
optimizer_actor = optim.Adam(actor_network.parameters(), lr=LR_actor)
optimizer_critic = optim.Adam(critic_network.parameters(), lr=LR_critic)

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# Agent initialization
agent = RandomAgent(m)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset environment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    ### Create Experience replay buffer ###
    buffer = ExperienceReplayBuffer(maximum_length=L)
    while not done:
        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32)
        mu, sigma2 = actor_network(state_tensor)

        mu = mu.detach().numpy()[0]
        sigma2 = sigma2.detach().numpy()[0]
        action = [np.random.normal(mu[0], np.sqrt(sigma2[0])), np.random.normal(mu[1], np.sqrt(sigma2[1]))]
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append experience to the buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # compute target value
    G_array = np.zeros(t)
    states, actions, rewards, next_states, dones = buffer.to_list()

    for idx in range(t):
        for jdx in range(idx, t):
            G_array[idx] += (discount_factor ** (jdx - idx)) * rewards[jdx]

    # compute advantage estimation
    states_tensor = torch.tensor(states,
                                requires_grad=True,
                                dtype=torch.float32)
    G_tensor = torch.tensor(G_array,
                                requires_grad=False,
                                dtype=torch.float32)
    mu_tensor, sigma2_tensor = actor_network(states_tensor)
    mu_tensor = mu_tensor.detach()
    sigma2_tensor = sigma2_tensor.detach()
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    pi_old_tensor = (compute_pi_probability(mu_tensor[:, 0], sigma2_tensor[:, 0], actions_tensor[:, 0]) *
                compute_pi_probability(mu_tensor[:, 1], sigma2_tensor[:, 1], actions_tensor[:, 1])).reshape(-1, )

    for n in range(M):
        train_critic_network(G_tensor, states_tensor)
        # update phi
        value = torch.reshape(critic_network(states_tensor), (-1,))
        phi_tensor = G_tensor - value
        train_actor_network(states_tensor, actions_tensor, phi_tensor, pi_old_tensor)

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    # stop training if reached a threshold
    #if running_average(episode_reward_list, n_ep_running_average)[-1] >= 150:
    #    break

#torch.save(actor_network, 'neural-network-3-actor.pth')
#torch.save(critic_network, 'neural-network-3-critic.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
