import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau
import get_Standarized_data

model = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[512, 512, 512, 512, 512])
print(model)
#data = getData_class.getData(['C'])
standarized_data = get_Standarized_data.get_Standarized_data("cScaler",'C')
print(standarized_data)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
torch.cuda.empty_cache()
training_class.train_model(model, loss, optimizer, x_train=standarized_data[0], y_train=standarized_data[1], x_test=standarized_data[2].squeeze(), y_test=standarized_data[3], save_path='modelCstandard_128_64_64_32.pth')

