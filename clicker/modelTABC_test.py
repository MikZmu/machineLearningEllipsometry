import os
import random

import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc0 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc0(out)
        return out


# Hyperparameters
input_size = 7
hidden_size = 50
output_size = 4
learning_rate = 0.001

# Create the model
model = MLP(input_size, hidden_size, output_size)

model.load_state_dict(torch.load("modelTABC.pth"))

model.eval()

project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder,"datasets", "Si_jaw_delta", "")
print(folder_path)
os.makedirs(folder_path, exist_ok=True)
# Get a list of all items in the folder
all_items = os.listdir(folder_path)
files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

random_file = random.choice(files)
print(random_file)
dataHelper = pd.read_csv(folder_path + random_file, sep='\t', header=None, index_col=False)
dataHelper = dataHelper.drop(index=[0])
dataHelper = dataHelper.drop(columns=[7])
dataHelper.columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
print(dataHelper)
x = dataHelper[['wavelength','psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']]
x = torch.from_numpy(x.values).float()
values = []
for i in x:
    with torch.no_grad():
        values.append(model(i))

T = 0
A = 0
B = 0
C = 0

for i in values:
    T = T + i[0].item()
    A = A + i[1].item()
    B = B + i[2].item()
    C = C + i[3].item()

print(f'T: {T/len(values)}')
print(f'A: {A/len(values)}')
print(f'B: {B/len(values)}')
print(f'C: {C/len(values)}')