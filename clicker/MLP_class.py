import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn=nn.ReLU):
        """
        Initialize the MLP model.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_layers (list): A list containing the number of neurons in each hidden layer (up to 5 layers).
            activation_fn (callable): Activation function to use (default is nn.ReLU).
        """
        super(MLP, self).__init__()
        if len(hidden_layers) > 5:
            raise ValueError("The number of hidden layers cannot exceed 5.")

        self.layers = nn.ModuleList()
        prev_size = input_size

        # Create hidden layers
        for neurons in hidden_layers:
            self.layers.append(nn.Linear(prev_size, neurons))
            self.layers.append(activation_fn())  # Add the specified activation function
            prev_size = neurons

        # Create output layer
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


module = MLP(input_size=7, output_size=1, hidden_layers=[32, 16, 16, 8])
print(module)