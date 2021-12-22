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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent
from DDPG_network import ActorNetwork
from DDPG_network import CriticNetwork
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from DDPG_soft_updates import soft_updates


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
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

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]
        # batch.append(new_exp)

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


def train_actor_network(exp):
    # Sample a batch of n elements & add latest
    states, actions, rewards, next_states, dones = buffer.sample_batch(
        N)
    pis = actor_network(torch.tensor(states, requires_grad=True))
    # pis = torch.reshape(pis, (-1,))

    # Training process, set gradients to 0
    optimizer_actor.zero_grad()

    Q = critic_network(torch.tensor(states, requires_grad=True), pis)
    Q = torch.reshape(Q, (-1,))

    # Compute gradient
    loss = (Q.sum()) * (-1 / N)
    loss.backward()

    # Clip gradient norm to 1
    nn.utils.clip_grad_norm_(actor_network.parameters(), max_norm=1)

    # Perform backward pass (backpropagation)
    optimizer_actor.step()


def train_critic_network(exp):
    # Sample a batch of n elements & add latest
    states, actions, rewards, next_states, dones = buffer.sample_batch(
        N)

    next_states_tensor = torch.tensor(next_states, requires_grad=True)
    pis = tar_actor_network(next_states_tensor)
    # pis = torch.reshape(pis, (-1,))
    # Training process, set gradients to 0
    optimizer_critic.zero_grad()

    targets_next = tar_critic_network(next_states_tensor, pis)
    targets_next = torch.reshape(targets_next, (-1,))
    y = torch.tensor(rewards, dtype=torch.float32, requires_grad=True) + discount_factor * targets_next * \
        (1 - torch.tensor(dones, dtype=torch.int32))

    Qa = critic_network(torch.tensor(states, dtype=torch.float32, requires_grad=True),
                        torch.tensor(actions, dtype=torch.float32, requires_grad=True))
    Qa = torch.reshape(Qa, (-1,))

    loss = nn.functional.mse_loss(Qa, y)
    # Compute gradient
    loss.backward()

    # Clip gradient norm to 1
    nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=1)

    # Perform backward pass (backpropagation)
    optimizer_critic.step()


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# initialize buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# Parameters
N_episodes = 300  # Number of episodes to run for training
discount_factor = 0.99  # Value of gamma
#discount_factor = 0.5  # Value of gamma
n_ep_running_average = 50  # Running average of 50 episodes
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality

#L = 30000  # buffer size
L = 200000  # buffer size
N = 64  # batch size
d = 2  # network is updated after every d steps
tau = 1e-3
LR_actor = 5e-5
LR_critic = 5e-4

# np.random.normal(mu, sigma, 1000)
mu = 0.15
sigma = 0.2

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

### Create Experience replay buffer ###
buffer = ExperienceReplayBuffer(maximum_length=L)

### Create networks ###
actor_network = ActorNetwork(state_size=dim_state, action_size=m)
tar_actor_network = copy.deepcopy(actor_network)
critic_network = CriticNetwork(state_size=dim_state, action_size=m)
tar_critic_network = copy.deepcopy(critic_network)

### Create optimizer ###
optimizer_actor = optim.Adam(actor_network.parameters(), lr=LR_actor)
optimizer_critic = optim.Adam(critic_network.parameters(), lr=LR_critic)

# Agent initialization
agent = RandomAgent(m)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# fill buffer with random experiences
# Reset enviroment data and initialize variables
while len(buffer) < L:
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        # Take a random action
        action = agent.forward(state)
        next_state, reward, done, _ = env.step(action)

        # Append experience to the buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    n_t = 0
    while not done:
        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32)

        # Compute output of the network
        noise_tensor = torch.tensor(n_t,
                                    requires_grad=False,
                                    dtype=torch.float32)
        action = actor_network(state_tensor) + noise_tensor

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action.detach().numpy()[0])

        # Append experience to the buffer
        exp = Experience(state, action.detach().numpy()[0], reward, next_state, done)
        buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        train_critic_network(exp)
        if t % d == 0:
            train_actor_network(exp)
            # soft update
            tar_critic_network = soft_updates(critic_network, tar_critic_network, tau)
            tar_actor_network = soft_updates(actor_network, tar_actor_network, tau)

        # Update state for next iteration
        state = next_state
        t += 1

        # Update n_t
        n_t = -mu * n_t + np.random.normal(mu, sigma, m)

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

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

#torch.save(actor_network, 'neural-network-2-actor.pth')
#torch.save(critic_network, 'neural-network-2-critic.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
