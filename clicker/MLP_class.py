import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn=nn.ReLU):

        super(MLP, self).__init__()
        if len(hidden_layers) > 7:
            raise ValueError("The number of hidden layers cannot exceed 7.")

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


class ConvMLP(nn.Module):
    def __init__(self, input_channels, output_size, conv_layers, fc_layers, activation_fn=nn.ReLU):
        """
        Constructs a Conv2d-based MLP for regression.

        Args:
            input_channels (int): Number of input channels for Conv2d.
            output_size (int): Number of output features (regression target size).
            conv_layers (list of tuples): Each tuple specifies (out_channels, kernel_size, stride, padding).
            fc_layers (list of int): Number of neurons in each fully connected layer.
            activation_fn (nn.Module): Activation function to use (default: nn.ReLU).
        """
        super(ConvMLP, self).__init__()

        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels

        # Create convolutional layers
        for out_channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(activation_fn())  # Add activation function
            prev_channels = out_channels

        self.flatten = nn.Flatten()

        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = None  # To be determined dynamically after flattening
        for neurons in fc_layers:
            if prev_size is not None:
                self.fc_layers.append(nn.Linear(prev_size, neurons))
            prev_size = neurons
            self.fc_layers.append(activation_fn())  # Add activation function

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the output
        x = self.flatten(x)

        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        # Output layer
        x = self.output_layer(x)
        return x