import os
import joblib
import pandas as pd
import torch
import statistics
from pandas.errors import CSSWarning
from scipy.stats import pearsonr

import MLP_class
from scipy import stats
from sklearn.metrics import mean_squared_error
import predict_class
import getData_class


#This function decodes the model name to extract number of the hidden layers

def decode_model(model):
    model = model.removesuffix(".pth")
    layers = model.split('_')
    layers.pop(0)
    return list(map(int, layers))

#This function loads the model from the specified path and returns it

def create_and_load(model_path, input_size, output_size):
    model = MLP_class.MLP(input_size, output_size, hidden_layers=decode_model(model_path))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def list_to_float(lst):
    return [float(i) for i in lst]

modelTT = create_and_load("modelT_48_32_16_8.pth",7,1)
modelAA = create_and_load("modelA_48_32_16_8.pth",7,1)
modelBB = create_and_load("modelB_256_256_256_128_64.pth",7,1)
modelCC = create_and_load("modelC_64_32_32_16.pth",7,1)
modelBBstandard = create_and_load("modelBstandard_64_32_32_16.pth",7,1)
modelCCstandard = create_and_load("modelCstandard_128_64_64_32.pth",7,1)

bScaler = joblib.load("bScaler_featureScaler.pkl")
bTargetScaler = joblib.load("bScaler_targetScaler.pkl")
cTargetScaler = joblib.load("cScaler_targetScaler.pkl")
cScaler = joblib.load("cScaler_featureScaler.pkl")

folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(folder,"datasets", "new_Si_jaw_delta", "")

data = getData_class.get_data_chunks(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["T", "A", "B", "C"], folder_path)
dataBStandard = getData_class.get_data_chunks(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["B"], folder_path)
dataCStandard = getData_class.get_data_chunks(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["C"], folder_path)


Tmeans = []
Treals = []

Ameans = []
Areals = []

Bmeans = []
Breals = []

Cmeans = []
Creals = []

BSmeans = []
BSreals = []

CSmeans = []
CSreals = []


for i in data:

    Tmeans.append(predict_class.predict_file_mean(modelTT,i[0]))
    Treals.append(i[1][0])

    Ameans.append(predict_class.predict_file_mean(modelAA,i[0]))
    Areals.append(i[1][1])

    Bmeans.append(predict_class.predict_file_mean(modelBB,i[0]))
    Breals.append(i[1][2])

    Cmeans.append(predict_class.predict_file_mean(modelCC,i[0]))
    Creals.append(i[1][3])

for i in dataBStandard:

    BSmeans.append(predict_class.predict_file_mean(modelBBstandard,i[0]))
    BSreals.append(i[1][0])

for i in dataCStandard[0]:
    CSmeans.append(predict_class.predict_file_mean(modelCCstandard,i[0]))
    CSreals.append(i[1][0])

Tmeans = list_to_float(Tmeans)
Treals = list_to_float(Treals)
Ameans = list_to_float(Ameans)
Areals = list_to_float(Areals)
Bmeans = list_to_float(Bmeans)
Breals = list_to_float(Breals)
Cmeans = list_to_float(Cmeans)
Creals = list_to_float(Cmeans)
BSmeans = list_to_float(BSmeans)
BSreals = list_to_float(BSreals)
CSmeans = list_to_float(CSmeans)
CSreals = list_to_float(CSreals)

rT = pearsonr(Tmeans, Treals)
rA = pearsonr(Ameans, Areals)
rB = pearsonr(Bmeans, Breals)
rC = pearsonr(Cmeans, Creals)
rBstandard = pearsonr(BSmeans, BSreals)
rCstandard = pearsonr(CSmeans, CSreals)


print("R2 dla T: " + str((float(rT[0])**2)))
print("R2 dla A: " + str((float(rA[0])**2)))
print("R2 dla B: " + str((float(rB[0])**2)))
print("R2 dla C: " + str((float(rC[0])**2)))
print("R2 dla B standard: " + str((float(rBstandard[0])**2)))
print("R2 dla C standard: " + str((float(rCstandard[0])**2)))



