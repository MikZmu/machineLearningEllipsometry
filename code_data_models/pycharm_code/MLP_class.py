import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn=nn.LeakyReLU):

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
        super(ConvMLP, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels
        for out_channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size, stride, padding))
            self.conv_layers.append(activation_fn())
            prev_channels = out_channels

        self.flatten = nn.Flatten()

        # Dynamically calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 71, 7)  # Adjusted for single sample size
            for layer in self.conv_layers:
                dummy_input = layer(dummy_input)
            flattened_size = dummy_input.numel()

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = flattened_size
        for neurons in fc_layers:
            self.fc_layers.append(nn.Linear(prev_size, neurons))
            self.fc_layers.append(activation_fn())
            prev_size = neurons

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x