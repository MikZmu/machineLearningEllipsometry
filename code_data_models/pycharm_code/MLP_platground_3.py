import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau
import get_Standarized_data
import getDatawithT
import get_StandarizedWithTA

model = MLP_class.MLP(input_size=9, output_size=1, hidden_layers=[32, 16, 16, 8])
print(model)
#data = getDatawithT.getData(['C'])
#data = get_Standarized_data.get_Standarized_data("cScaler",'C')
data = get_StandarizedWithTA.get_Standarized_data('CTAscaler', ['C'])
#print(data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
torch.cuda.empty_cache()
training_class.train_model(model, loss, optimizer, x_train=data[0], y_train=data[1], x_test=data[2].squeeze(), y_test=data[3], save_path='modelCstandardwithTA_32_16_16_8.pth')

