import os
import random
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy import stats
import MLP_class


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
output_size = 4
learning_rate = 0.001

# Create the model
model = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])

model.load_state_dict(torch.load("modelB.pth"))

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
info = random_file.split('_')
T = info[0]
A = info[1]
B = info[2]
C = info[3]
C = C.removesuffix(".txt")
if ("-" in C):
    C = C.removesuffix("e-")
    C = float(C) * 10 ** -5
elif ("e" in C):
    C = C.removesuffix("e")
    C = float(C) * 10 ** -5

if (float(C) > 1):
    C = float(C) * 10 ** -5

dataHelper = pd.read_csv(folder_path + random_file, sep='\t', header=None, index_col=False)
dataHelper = dataHelper.drop(index=[0])
dataHelper = dataHelper.drop(columns=[7])
dataHelper.columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
dataHelper['T'] = T
dataHelper['A'] = A
dataHelper['B'] = B
dataHelper['C'] = C
print(dataHelper)
x = dataHelper[['wavelength','psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']]
x = torch.from_numpy(x.values).float()
values = []

for i in x:
    with torch.no_grad():
        values.append(model(i).item())


T = 0

for i in values:
    T = T + i

print(f'T: {T/len(values)}')
reals = dataHelper["B"].tolist()
reals = list(map(float, reals))


for i in values:
    print(f'prediction: {i}, real: {reals[values.index(i)]}')

plt.figure(figsize=(10, 6))
plt.plot(values, label='Predicted Values', marker='o')
plt.plot(reals, label='Real Values', marker='x')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Comparison of Predicted and Real Values')
plt.legend()
plt.grid(True)
plt.show()