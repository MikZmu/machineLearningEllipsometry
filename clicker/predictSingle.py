import os

import joblib

import MLP_class
import pandas as pd
import torch
from torch import layout
import statistics

project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder,"datasets", "new_Si_jaw_delta", "")
all_items = os.listdir(folder_path)
files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

def extract_from_name(name):
    dataHelper = pd.read_csv(folder_path + name, sep='\t', header=None, index_col=False)
    info = name.split('_')
    T = info[0]
    A = info[1]
    B = info[2]
    C = info[3]
    dataHelper = dataHelper.drop(index=[0])
    dataHelper = dataHelper.drop(columns=[7])
    dataHelper.columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
    dataHelper['T'] = T
    dataHelper['A'] = A
    dataHelper['B'] = B
    C = C.removesuffix(".txt")
    print('x')
    if ("-" in C):
        C = C.removesuffix("e-")
        C = float(C) * 10 ** -5
    elif ("e" in C):
        C = C.removesuffix("e")
        C = float(C) * 10 ** -5

    if (float(C) > 1):
        C = float(C) * 10 ** -5

    dataHelper['C'] = C

    return dataHelper

def decode_model(model):
    model = model.removesuffix(".pth")
    layers = model.split('_')
    layers.pop(0)
    return list(map(int, layers))




def predict(file, model):
    print(file)
    print(model)
    dataHelper = extract_from_name(file)
    x = dataHelper[['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']]
    x = torch.from_numpy(x.values).float()
    layers = decode_model(model)
    print(layers)
    mModel = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=layers)
    if torch.cuda.is_available():
        mModel.load_state_dict(torch.load(model))
    else:
        mModel.load_state_dict(torch.load(model, map_location=torch.device('cpu')))

    values = []

    for i in x:
        with torch.no_grad():
            values.append(mModel(i).item())

    print(f"Model prediction: {statistics.median(values)}" )



def predict_scaled(file, model, scaler, targetscaler):
    print(file)
    print(model)
    print(scaler)
    print(targetscaler)
    dataHelper = extract_from_name(file)
    x = dataHelper[['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']]
    x = torch.from_numpy(x.values).float()
    layers = decode_model(model)
    print(layers)
    mModel = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=layers)
    if torch.cuda.is_available():
        mModel.load_state_dict(torch.load(model))
    else:
        mModel.load_state_dict(torch.load(model, map_location=torch.device('cpu')))

    featureNames = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
    featureScaler = joblib.load(scaler)
    targetScaler = joblib.load(targetscaler)
    values = []
    for i in x:
        i_df = pd.DataFrame(i.numpy().reshape(1, -1), columns=featureNames)
        iScaled = featureScaler.transform(i_df)
        with torch.no_grad():
            pred = mModel(torch.from_numpy(iScaled).float())
            values.append(pred.item())

    print(f"Model prediction: {statistics.median(values)}")






predict("94.848_1.3902_0.01375279_0.00019072660000000002.txt", "modelT_48_32_16_8.pth")

predict_scaled("94.848_1.3902_0.01375279_0.00019072660000000002.txt", "modelBstandard_96_48_24_12.pth", "bScaler_featureScaler.pkl", "bScaler_targetScaler.pkl")