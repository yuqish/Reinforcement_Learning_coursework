import torch
import torch.nn as nn
import torch.optim as optim

### Neural Network ###
class ActorNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, state_size, action_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer1 = nn.Linear(state_size, 400)
        self.input_layer_activation1 = nn.ReLU()
        self.input_layer2 = nn.Linear(400, 200)
        self.input_layer_activation2 = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(200, action_size)
        self.output_layer_activation = nn.Tanh()

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer1(x)
        l1 = self.input_layer_activation1(l1)

        # Compute second layer
        l1 = self.input_layer2(l1)
        l1 = self.input_layer_activation2(l1)

        # Compute output layer
        l1 = self.output_layer(l1)
        out = self.output_layer_activation(l1)
        return out

class CriticNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, state_size, action_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer1 = nn.Linear(state_size, 400)
        self.input_layer_activation1 = nn.ReLU()
        self.input_layer2 = nn.Linear(400 + action_size, 200)
        self.input_layer_activation2 = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(200, 1)

    def forward(self, s, a):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer1(s)
        l1 = self.input_layer_activation1(l1)

        # Compute second layer
        l1 = self.input_layer2(torch.cat([l1, a], dim=1))
        l1 = self.input_layer_activation2(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out