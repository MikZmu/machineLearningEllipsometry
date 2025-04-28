import os
import joblib
import MLP_class
import pandas as pd
import torch
import statistics



def extract_from_name(name, folder_path):
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
