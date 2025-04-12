import torch

import getData_class
import MLP_class
data = getData_class.getData()

x_train, y_train, x_test, y_test = data

random_index = torch.randint(0, x_test.shape[0], (1,))
random_x = x_train[random_index]
random_y = y_train[random_index]

model = MLP_class.MLP(input_size=7, output_size=4, hidden_layers=[ 64, 32, 32, 16])

model.load_state_dict(torch.load('modelT.pth'))

model.eval()

print(f"Input: {random_x}")
print(f"Expected Output: {random_y}")
with torch.no_grad():
    prediction = model(random_x)
print(f"Prediction: {prediction}")