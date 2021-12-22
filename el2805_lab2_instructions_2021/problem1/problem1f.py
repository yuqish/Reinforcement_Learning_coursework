import numpy as np

from DQN_network import MyNetwork
import gym
from DQN_agent import RandomAgent
import torch
import math
import matplotlib.pyplot as plt


nn = torch.load('neural-network-1.pth')
y_array = np.linspace(0, 1.5, 100)
omega_array = np.linspace(-math.pi, math.pi, 100)
action_array = np.zeros((len(y_array), len(omega_array)))
Qmax_array = np.zeros((len(y_array), len(omega_array)))

for i in range(len(y_array)):
    for j in range(len(omega_array)):
        y = y_array[i]
        omega = omega_array[j]
        # Take an action according to nn
        state_tensor = torch.tensor([[0, y, 0, 0, omega, 0, 0, 0]],
                                    requires_grad=False,
                                    dtype=torch.float32)
        x = nn(state_tensor)
        Qmax_array[i, j] = x.max(1)[0].item()
        action_array[i, j] = x.max(1)[1].item()


plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(omega_array, y_array, Qmax_array, 100, cmap='binary')
ax.set_xlabel('omega')
ax.set_ylabel('y')
ax.set_zlabel('Q')

# contour plot doesn't look good for actions
y_array_new = []
omega_array_new = []
action_array_new = []
Qmax_array_new = []
for i in range(len(y_array)):
    for j in range(len(omega_array)):
        y_array_new.append(y_array[i])
        omega_array_new.append(omega_array[j])
        action_array_new.append(action_array[i,j])
        Qmax_array_new.append(Qmax_array[i,j])
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(y_array_new, omega_array_new, action_array_new)
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('action')


plt.show()

