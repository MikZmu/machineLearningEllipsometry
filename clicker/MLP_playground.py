import torch
from torch import nn
import MLP_class
import  getData_class
import training_class


model = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[32, 16, 16, 8])
print(model)
data = getData_class.getData()
print(data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_class.train_model(model, loss, optimizer, x_train=data[0], y_train=data[1], x_test=data[2], y_test=data[3], save_path='modelT.pth')
