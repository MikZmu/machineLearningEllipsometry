import time

import epoch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from scipy.signal import find_peaks
import os
import numpy as np
import pandas as pd
from IPython.display import clear_output

project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder,"datasets", "Si_jaw_delta", "")
print(folder_path)
os.makedirs(folder_path, exist_ok=True)
# Get a list of all items in the folder
all_items = os.listdir(folder_path)
print(all_items)

# Filter the list to include only files
files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

dataFrame = pd.DataFrame()

newDataFrame = pd.DataFrame()

for i in files:
    dataHelper = pd.read_csv(folder_path + i, sep='\t', header=None, index_col=False)
    info = i.split('_')
    T = info[0]
    A = info[1]
    B = info[2]
    C = info[3]
    dataHelper = dataHelper.drop(index=[0])
    dataHelper = dataHelper.drop(columns=[7])
    dataHelper.columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
    dataHelper['T'] = T
    dataHelper['A'] = A
    dataHelper['B'] = B
    C = C.removesuffix(".txt")

    if ("-" in C):
        C = C.removesuffix("e-")
        C = float(C) * 10 ** -5
    elif ("e" in C):
        C = C.removesuffix("e")
        C = float(C) * 10 ** -5

    if (float(C) > 1):
        C = float(C) * 10 ** -5

    dataHelper['C'] = C
    dataFrame = pd.concat([dataFrame, dataHelper], ignore_index=True)

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(7, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc0 = nn.Linear(8, 1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc0(out)
        return out


# Hyperparameters
input_size = 7
hidden_size = 50
output_size = 1
learning_rate = 0.002

# Create the model
model = MLP(input_size, hidden_size, output_size)


# Loss and optimizer
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)
    ss_residual = torch.sum((target - output) ** 2)
    r2 = 1 - ss_residual / ss_total
    return 1 - r2  # We return 1 - R² because we want to minimize the loss

crit = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data (replace with your actual data)
x_train, x_test , y_train, y_test = train_test_split(dataFrame[['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']], dataFrame[['T']], test_size=0.2, random_state=42)

x_train = torch.from_numpy(x_train.values).float()
x_test = torch.from_numpy(x_test.values).float()
y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))
import matplotlib as plt
from IPython.display import clear_output
start_time = time.time()
loss = 100000000
bestLoss = 10000000
# Training loop
while True:
    # Forward pass
    outputs = model(x_train)
    loss = crit(outputs, y_train)
    testLoss = crit(model(x_test), y_test)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    times = []
    losses = []
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    losses.append(loss.item())

    # Update the graph
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(times, losses, label="Training Loss")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title("Loss vs Time")
    plt.legend()
    plt.grid()
    plt.show()
    print(f'Loss: {loss.item():.4f}')
    print(f'Test loss: {testLoss.item():.4f}')

    if loss < bestLoss and testLoss<bestLoss*1.5:
        bestLoss = loss
        torch.save(model.state_dict(), 'modelT_7x32x16x16x8x1.pth')
