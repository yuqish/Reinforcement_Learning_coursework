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

        #transition 1
        self.input_trans1 = nn.Linear(400, 200)
        #transition 2
        self.input_trans2 = nn.Linear(400, 200)

        # Create output layer
        self.output_layer_head1 = nn.Linear(200, action_size)
        self.output_layer_head1_activation = nn.Tanh()
        self.output_layer_head2 = nn.Linear(200, action_size)
        self.output_layer_head2_activation = nn.Sigmoid()

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        x = self.input_layer1(x)
        x = self.input_layer_activation1(x)

        # mu
        # Compute second layer
        x1 = self.input_trans1(x)

        # Compute output layer
        x1 = self.output_layer_head1(x1)
        out1 = self.output_layer_head1_activation(x1)

        # sigma^2
        # Compute second layer
        x2 = self.input_trans2(x)

        # Compute output layer
        x2 = self.output_layer_head2(x2)
        out2 = self.output_layer_head2_activation(x2)
        return out1, out2

class CriticNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, state_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer1 = nn.Linear(state_size, 400)
        self.input_layer_activation1 = nn.ReLU()
        self.input_layer2 = nn.Linear(400, 200)
        self.input_layer_activation2 = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(200, 1)

    def forward(self, s):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer1(s)
        l1 = self.input_layer_activation1(l1)

        # Compute second layer
        l1 = self.input_layer2(l1)
        l1 = self.input_layer_activation2(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out