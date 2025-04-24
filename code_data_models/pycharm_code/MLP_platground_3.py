import os.path
import torch
from torch import nn
import MLP_class
import  getData_class
import training_class
from torch.optim.lr_scheduler import ReduceLROnPlateau
import get_Standarized_data

#data = getDatawithT.getData(['C'])
#data = get_Standarized_data.get_Standarized_data("cScaler",'C')
path = os.path.dirname(os.path.abspath(__file__))
pf = os.path.join(path, "..", "datasets", "new_Si_jaw_delta", "")
data = getData_class.get_conv_data(['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], ['C'], pf)
model = MLP_class.ConvMLP(input_channels=1, output_size=1,conv_layers = [(16,3,1,1), (32,3,1,1)] , fc_layers =[128, 64])
training_class.train_conv_mlp(model,
                              loss_fn=nn.MSELoss(),
                              optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                              x_train=data[0][0],
                              y_train=data[0][1],
                              x_val=data[1][0],
                              y_val=data[1][1],
                              save_path="modelC.pth",
                              patience=1000000000000000,
                              max_epochs=100000)
print(model)
