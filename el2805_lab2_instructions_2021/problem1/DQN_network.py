import torch
import torch.nn as nn
import torch.optim as optim

### Neural Network ###
class MyNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer1 = nn.Linear(input_size, 64)
        self.input_layer_activation1 = nn.ReLU()
        self.input_layer2 = nn.Linear(64, 64)
        self.input_layer_activation2 = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(64, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer1(x)
        l1 = self.input_layer_activation1(l1)

        # Compute second layer
        l1 = self.input_layer2(l1)
        l1 = self.input_layer_activation2(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out