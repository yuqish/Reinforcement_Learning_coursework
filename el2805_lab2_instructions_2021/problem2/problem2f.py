import numpy as np

from DDPG_network import ActorNetwork
from DDPG_network import CriticNetwork
import gym
import torch
import math
import matplotlib.pyplot as plt


nn_actor = torch.load('neural-network-2-actor.pth')
nn_critic = torch.load('neural-network-2-critic.pth')
y_array = np.linspace(0, 1.5, 100)
omega_array = np.linspace(-math.pi, math.pi, 100)
action_angle_array = np.zeros((len(y_array), len(omega_array)))
Q_array = np.zeros((len(y_array), len(omega_array)))

for i in range(len(y_array)):
    for j in range(len(omega_array)):
        y = y_array[i]
        omega = omega_array[j]
        # Take an action according to nn
        state_tensor = torch.tensor([[0, y, 0, 0, omega, 0, 0, 0]],
                                    requires_grad=False,
                                    dtype=torch.float32)
        action = nn_actor(state_tensor)
        Q = nn_critic(state_tensor, action)
        Q_array[i, j] = Q.detach().numpy()[0][0]
        action = action.detach().numpy()[0]
        action_angle_array[i, j] = action[1]


plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(omega_array, y_array, Q_array, 100, cmap='binary')
ax.set_xlabel('omega')
ax.set_ylabel('y')
ax.set_zlabel('Q')

# contour plot doesn't look good for actions
y_array_new = []
omega_array_new = []
action_angle_array_new = []
Q_array_new = []
for i in range(len(y_array)):
    for j in range(len(omega_array)):
        y_array_new.append(y_array[i])
        omega_array_new.append(omega_array[j])
        action_angle_array_new.append(action_angle_array[i,j])
        Q_array_new.append(Q_array[i,j])
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y_array_new, omega_array_new, action_angle_array_new)
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('action')


plt.show()

