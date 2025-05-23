import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau
import get_Standarized_data

model = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[32, 16, 16, 8])
print(model)
data = getData_class.getData(['B'])
#standarized_data = get_Standarized_data.get_Standarized_data("bScaler",['B'])
#print(standarized_data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
torch.cuda.empty_cache()
training_class.train_model(model, loss, optimizer, x_train=data[0], y_train=data[1], x_test=data[2].squeeze(), y_test=data[3], save_path='modelB_32_16_16_8.pth')
#training_class.train_model(model, loss, optimizer, x_train=standarized_data[0], y_train=standarized_data[1], x_test=standarized_data[2], y_test=standarized_data[3], save_path='modelBstandard_64_48_48_24.pth')

