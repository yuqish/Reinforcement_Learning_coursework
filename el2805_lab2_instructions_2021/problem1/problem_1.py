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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from collections import deque, namedtuple
import copy
from DQN_network import MyNetwork

# initialize buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()


# Parameters
N_episodes = 500                             # Number of episodes
#N_episodes = 250                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
#discount_factor = 1                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
epsilon = 0
epsilon_min = 0.05
epsilon_max = 1

L = 30000   # buffer size
#L = 5000   # buffer size
N = 128   # batch size
#C = 5
C = L/N    # network is updated after every C steps
LR = 0.0008


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

    def sample_batch(self, n, new_exp):
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
        batch.append(new_exp)

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)



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

def train_network(exp, losses):
    # Sample a batch of n elements & add latest
    states, actions, rewards, next_states, dones = buffer.sample_batch(
        N-1, exp)

    # Training process, set gradients to 0
    optimizer.zero_grad()

    targets_next = tar_network(torch.tensor(next_states)).detach().max(1)[0].unsqueeze(1)
    targets_next = torch.reshape(targets_next, (-1,))
    y = torch.tensor(rewards, dtype=torch.float32) + discount_factor * targets_next * (1 - torch.tensor(dones, dtype=torch.int32))

    Qa = network(torch.tensor(states, requires_grad=True))
    Qa = torch.gather(Qa, 1, torch.tensor([[a] for a in actions]).long())
    Qa = torch.reshape(Qa, (-1,))
    loss = nn.functional.mse_loss(Qa, y)
    losses.append(loss.item())
    # Compute gradient
    loss.backward()

    # Clip gradient norm to 1
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)

    # Perform backward pass (backpropagation)
    optimizer.step()

def update_network():
    tar_network = copy.deepcopy(network)
    return tar_network

def epsilon_greedy(values, n_actions, ep):
    action_probabilities = np.ones(n_actions, dtype=float) * ep / n_actions
    best_action = values.max(1)[1].item()
    action_probabilities[best_action] += (1.0 - ep)
    action = np.random.choice(np.arange(
        len(action_probabilities)),
        p=action_probabilities)
    return action

def decay_epsilon(k, Z):
    # exponential
    #epsilon = max([epsilon_min, epsilon_max*((epsilon_min/epsilon_max)**((k-1)/(Z-1)))])
    # linear
    epsilon = max([epsilon_min, epsilon_max - ((epsilon_max - epsilon_min) * (k - 1) / (Z - 1))])
    return epsilon



### Create Experience replay buffer ###
buffer = ExperienceReplayBuffer(maximum_length=L)

### Create network ###
network = MyNetwork(input_size=dim_state, output_size=n_actions)
tar_network = copy.deepcopy(network)

### Create optimizer ###
optimizer = optim.Adam(network.parameters(), lr=LR)

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# fill buffer with random experiences
# Reset enviroment data and initialize variables
for i in range(2):
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

t_step = 0
ave_losses = []
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    epsilon = decay_epsilon(i+1, 0.9*N_episodes)
    t = 0
    # debug
    losses = []
    while not done:
        # Take a random action
        #action = agent.forward(state)

        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state],
                                    requires_grad=False,
                                    dtype=torch.float32)

        # Compute output of the network
        values = network(state_tensor)

        # Pick the action with greatest value
        # .max(1) picks the action with maximum value along the first dimension
        # [1] picks the argmax
        # .item() is used to cast the tensor to a real value
        action = epsilon_greedy(values, n_actions, epsilon)


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
        t_step+= 1

        train_network(exp, losses)
        if t_step % C == 0:
            tar_network = update_network()

    ave_losses.append(sum(losses)/len(losses))
    # Append episode reward and total number of steps
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

print('average reward: ', running_average(episode_reward_list, n_ep_running_average)[-1])
print('average step: ', running_average(episode_number_of_steps, n_ep_running_average)[-1])


#torch.save(network, 'neural-network-1.pth')

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
#plt.show()

plt.figure()
plt.plot([i for i in range(1, N_episodes+1)], ave_losses)


plt.show()

