import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau
import get_Standarized_data

model = MLP_class.ConvMLP(input_channels=7, output_size=4, conv_layers=[(16, 3, 1, 1), (32, 3, 1, 1)], fc_layers=[128, 64], activation_fn=nn.ReLU)
print(model)
data = getData_class.getData()

#print(data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
torch.cuda.empty_cache()
training_class.train_conv_mlp(model, loss, optimizer, x_train=data[0], y_train=data[1], x_test=data[2].squeeze(), y_test=data[3], save_path='modelCstandard_256_128_64_32.pth')

