import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[1024, 1024, 1024, 1024, 1024,1024,1024])
print(model)
data = getData_class.getData(['B'])
print(data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
torch.cuda.empty_cache()
training_class.train_model(model, loss, optimizer, x_train=data[0], y_train=data[1], x_test=data[2].squeeze(), y_test=data[3], save_path='modelB_1024_1024_1024_1024_1024_1024_1024.pth')

