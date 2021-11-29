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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import math
import sys
from matplotlib import cm


#log
#sys.stdout = open('log', 'w')

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 200        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma
epsilon = 0
T = 200
#eta = np.array([[0,0],[1,0],[0,1],[1,1],[1,2],[2,1]])
eta = np.array([[1,0],[0,1],[1,1],[1,2],[2,1]])
m = np.shape(eta)[0]
lambda0 = 0
alpha = 0.1
objective_reward = -135

# scale learning rate
def scale_learning_rate(alpha,eta):
    alpha_scaled = np.zeros(m)
    for l in range(m):
        if np.linalg.norm(eta[l,:],ord=2) != 0:
            alpha_i = alpha/np.linalg.norm(eta[l,:],ord=2)
        else:
            alpha_i = alpha
        alpha_scaled[l] = alpha_i
    return alpha_scaled





# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


def get_action(Q, ep):
    action_probabilities = np.ones(k, dtype=float) * ep / k
    best_action = np.argmax(Q)
    action_probabilities[best_action] += (1.0 - ep)
    action = np.random.choice(np.arange(
        len(action_probabilities)),
        p=action_probabilities)
    return action


def CalculateQa(w, basis, action):
    Qa = np.dot(np.transpose(w[:,action]),basis)
    return Qa

def traditional_w(w, delta, alpha_scaled, z):
    for a in range(k):
        w[:, a] = w[:, a] + delta * np.multiply(alpha_scaled, z[:, a])
        # w[:, a] = w[:, a] + alpha*delta*z[:,a]
    return w

def momentum_w(w, delta, alpha_scaled, z, v):
    m0 = 0.8
    for a in range(k):
        v[:,a] = m0*v[:,a] + delta * np.multiply(alpha_scaled, z[:, a])
        w[:, a] = w[:, a] + v[:, a]
    return (w,v)

def nesterov_w(w, delta, alpha_scaled, z, v):
    m0 = 0.8
    for a in range(k):
        v[:,a] = m0*v[:,a] + delta * np.multiply(alpha_scaled, z[:, a])
        w[:, a] = w[:, a] + m0*v[:, a] + delta * np.multiply(alpha_scaled, z[:, a])
    return (w,v)

#initialize parameters
# Reward

w_init = np.array([[-39.26965632, -12.07199375, -18.39139687],\
 [-17.45133093, -40.84059995, -50.74872652],\
 [ 40.96576869,  48.60967175,  49.15348425],\
 [-11.05354125,  -0.7282752,    9.28469829],\
 [ -9.57619056,  -4.1354807,  -19.83631327]])

#w_init = np.zeros((m,k))
v = np.zeros((m,k))
def train_network(w_init,v,alpha,eta,lambda0,N_episodes):
    w = np.copy(w_init)
    episode_reward_list = []  # Used to save episodes reward
    alpha_scaled = scale_learning_rate(alpha, eta)
    # Training process
    for i in range(N_episodes):
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset())
        total_episode_reward = 0

        z = np.zeros((m, k))
        t = 0   # t is actually not needed since when t=200 'done' becomes True
        while not done:
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            basis = np.cos(math.pi * np.dot(eta, state))
            #print(eta)
            #print(state)

            Q = [CalculateQa(w,basis,action) for action in range(k)]
            #print(Q)
            action = get_action(Q,epsilon)
            #print(action)
            #action = np.random.randint(0, k)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            next_state = scale_state_variables(next_state)
            next_basis = np.cos(math.pi*np.dot(eta,next_state))
            next_Q = [CalculateQa(w, next_basis, action) for action in range(k)]
            next_action = get_action(next_Q,epsilon)

            # compute TD
            delta = reward + discount_factor*CalculateQa(w,next_basis,next_action) - CalculateQa(w,basis,action)

            # update z
            for a in range(k):
                if action == a:
                    z[:, action] = discount_factor*lambda0*z[:,action] + basis
                else:
                    z[:, a] = discount_factor*lambda0*z[:,a]

            # clipping eligibility trace
            z = np.clip(z, -5, 5)
            #print(z)

            # update w
            #w = traditional_w(w, delta, alpha_scaled, z)
            (w, v) = nesterov_w(w, delta, alpha_scaled, z, v)
            #print('w:',w)
            #print('v:',v)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t += 1
            if t >= T:
                break

        #print(total_episode_reward)
        #print('episode finished ', i)
        # decrease learning rate
        if total_episode_reward - objective_reward > -20:
            #converging
            #print('converging')
            alpha_scaled = alpha_scaled*(1-0.3)
            #print(alpha_scaled)

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        # Close environment
        #env.close()

    return (episode_reward_list, w)

print('question (b)')
(episode_reward_list, w) = train_network(w_init,v,alpha,eta,lambda0,N_episodes)
print('average in last 50 is: ', sum(episode_reward_list[-50:])/50)



print('question (d)(1)')
# Plot Rewards
plt.figure()
plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 10), label='Average episode reward')
plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 50), label='Average episode reward 50')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()
plt.grid(alpha=0.3)
#plt.show()

print('question (d)(2), (d)(3)')
# 3D plot of optimal policy and value function

# for test
#w = np.array([[ -48.63816811,  -18.00940319,  -10.02344002],
# [ -29.047999,    -36.75267136, -193.68386872],
# [  47.91239433,   57.83002841,   69.07181856],
# [ -17.22764305,   -0.99102807,  -23.84577267],
# [  -7.73921536,   -6.70650195,  -37.957208  ]])

#result from simulation
s0 = env.reset()
state = scale_state_variables(s0)
states_array = [s0]
rewards_array = []
optimal_action_array = []
done = False
total_reward = 0
while not done:
    basis = np.cos(math.pi * np.dot(eta, state))
    Q = [CalculateQa(w, basis, action) for action in range(k)]
    action = np.argmax(Q)
    optimal_action_array.append(action)
    next_state, reward, done, _ = env.step(action)
    states_array.append(next_state)
    rewards_array.append(reward)
    next_state = scale_state_variables(next_state)
    total_reward += reward
    state = next_state


rewards_array.reverse()
rewards_array_sum = list(np.cumsum(rewards_array))
rewards_array_sum.reverse()

states_array1 = [states_array[i][0] for i in range(len(states_array))]
states_array2 = [states_array[i][1] for i in range(len(states_array))]


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(states_array1[:-1], states_array2[:-1], rewards_array_sum, cmap='Greens');
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('Simulated V')


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(states_array1[:-1], states_array2[:-1], optimal_action_array, cmap='Greens');
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('Simulated Action')


# result from calculation
s1_array = np.linspace(-1.2, 0.6, num=101)
s2_array = np.linspace(-0.07, 0.07, num=101)
action_array = np.zeros((len(s1_array), len(s2_array)))
total_reward_array = np.zeros((len(s1_array), len(s2_array)))
for i in range(len(s1_array)):
    for j in range(len(s2_array)):
        state = [s1_array[i], s2_array[j]]
        state = scale_state_variables(state)
        basis = np.cos(math.pi * np.dot(eta, state))
        Q = [CalculateQa(w, basis, action) for action in range(k)]
        action = np.argmax(Q)
        V = np.max(Q)
        total_reward_array[i,j] = V
        action_array[i,j] = action

        #print('state:',[s1_array[i], s2_array[j]])
        #print('action:',action)

print(total_reward_array)
plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(s1_array, s2_array, total_reward_array, 50, cmap='binary')
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('V')

# contour plot doesn't look good for actions
s1_array_new = []
s2_array_new = []
action_array_new = []
for i in range(len(s1_array)):
    for j in range(len(s2_array)):
        s1_array_new.append(s1_array[i])
        s2_array_new.append(s2_array[j])
        action_array_new.append(action_array[i,j])
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(s1_array_new, s2_array_new, action_array_new)
ax.set_xlabel('s1')
ax.set_ylabel('s2')
ax.set_zlabel('Action')


print('question (d)(5)')
# episodic total reward plot for agent taking actions uniformly at random and my agent

# uniformly at random
episode_total_random = []

for i in range(50):
    # Reset enviroment data
    done = False
    state = scale_state_variables(env.reset())
    total_reward = 0
    while not done:
        basis = np.cos(math.pi * np.dot(eta, state))
        Q = [CalculateQa(w, basis, action) for action in range(k)]
        action = get_action(Q, 1) # get action for epsilon=1
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        total_reward += reward
        state = next_state
    episode_total_random.append(total_reward)

# my agent
episode_total_my_agent = []

for i in range(50):
    # Reset enviroment data
    done = False
    state = scale_state_variables(env.reset())
    total_reward = 0
    while not done:
        basis = np.cos(math.pi * np.dot(eta, state))
        Q = [CalculateQa(w, basis, action) for action in range(k)]
        action = np.argmax(Q)
        next_state, reward, done, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        total_reward += reward
        state = next_state
    episode_total_my_agent.append(total_reward)

print('episode total random: ', episode_total_random)
print('episode my agent: ', episode_total_my_agent)

plt.figure()
plt.plot(np.linspace(1,50,50),episode_total_random, label='Episode reward random')
plt.plot(np.linspace(1,50,50),episode_total_my_agent, label='Episode reward my agent')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Total Reward vs Episodes')
plt.legend()

print('question (e). See plots')
ave_total_varalpha = []
for alpha1 in np.linspace(0.05,0.5,10):
    (episode_reward_list, w_new) = train_network(w_init,v,alpha1,eta,lambda0,50)
    #print(episode_reward_list)
    #print(alpha1)
    #print(sum(episode_reward_list[-50:])/50)
    ave_total_varalpha.append(sum(episode_reward_list[-50:])/50)

plt.figure()
plt.plot(np.linspace(0.05,0.5,10),ave_total_varalpha)
plt.xlabel('Learning Rate')
plt.ylabel('Average Total Reward')
plt.title('Average Total Reward vs Learning Rate')

ave_total_varlambda = []
for lambda1 in np.linspace(0,1,11):
    (episode_reward_list, w_new) = train_network(w_init,v,alpha,eta,lambda1,50)
    #print(episode_reward_list)
    #print(lambda1)
    #print(sum(episode_reward_list[-50:])/50)
    ave_total_varlambda.append(sum(episode_reward_list[-50:])/50)

plt.figure()
plt.plot(np.linspace(0,1,11),ave_total_varlambda)
plt.xlabel('Lambda')
plt.ylabel('Average Total Reward')
plt.title('Average Total Reward vs Lambda')

plt.show()



#sys.stdout.close()