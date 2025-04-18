import os

import joblib
import pandas as pd
import torch
import statistics
import MLP_class
from scipy import stats
from sklearn.metrics import mean_squared_error



modelT = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[48, 32, 16, 8])
if torch.cuda.is_available():
    modelT.load_state_dict(torch.load("modelT_48_32_16_8.pth"))
else:
    modelT.load_state_dict(torch.load("modelT_48_32_16_8.pth", map_location=torch.device('cpu')))


modelA = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[48, 32, 16, 8])
if torch.cuda.is_available():
    modelA.load_state_dict(torch.load("modelA_48_32_16_8.pth"))
else:
    modelA.load_state_dict(torch.load("modelA_48_32_16_8.pth", map_location=torch.device('cpu')))

modelB = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[1024, 1024, 1024, 1024, 1024, 1024, 1024])

modelBstandard = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[64, 32, 32, 16])
modelCstandard = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])

modelC = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[1024, 1024, 1024, 1024, 1024, 1024, 1024])
if torch.cuda.is_available():
    #modelC.load_state_dict(torch.load("modelC_1024_1024_1024_1024_1024_1024_1024.pth"))
    #modelB.load_state_dict(torch.load("modelB_1024_1024_1024_1024_1024_1024_1024.pth"))
    modelBstandard.load_state_dict(torch.load("modelBstandard_64_32_32_16.pth"))
    modelCstandard.load_state_dict(torch.load("modelCstandard_128_64_64_32.pth"))
else:
    #modelC.load_state_dict(torch.load("modelC_1024_1024_1024_1024_1024_1024_1024.pth", map_location=torch.device('cpu')))
    #modelB.load_state_dict(torch.load("modelB_1024_1024_1024_1024_1024_1024_1024.pth", map_location=torch.device('cpu')))
    modelBstandard.load_state_dict(torch.load("modelBstandard_64_32_32_16.pth", map_location=torch.device('cpu')))
    modelCstandard.load_state_dict(torch.load("modelCstandard_128_64_64_32.pth", map_location=torch.device('cpu')))


bScaler = joblib.load("bScaler_featureScaler.pkl")
bTargetScaler = joblib.load("bScaler_targetScaler.pkl")
cTargetScaler = joblib.load("cScaler_targetScaler.pkl")
cScaler = joblib.load("cScaler_featureScaler.pkl")


project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder,"datasets", "new_Si_jaw_delta", "")
print(folder_path)
os.makedirs(folder_path, exist_ok=True)
all_items = os.listdir(folder_path)
print(all_items)

# Filter the list to include only files
files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

dataFrame = pd.DataFrame()

newDataFrame = pd.DataFrame()
modelTSetVal = []
realTSetVal = []
modelASetVal = []
realASetVal = []
modelBSetVal = []
realBSetVal = []
modelCSetVal = []
realCSetVal = []
modelBstandardSetVal = []
modelCstandardSetVal = []

for i in files:
        dataHelper = pd.read_csv(folder_path + i, sep='\t', header=None, index_col=False)
        info = i.split('_')
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

        x = dataHelper[['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']]
        x = torch.from_numpy(x.values).float()
        aVal = []
        tVal = []
        bVal = []
        cVal = []
        bStandardVal = []
        cStandardVal = []

        values = [[tVal,aVal,bVal,cVal]]

        for a in x:
            with torch.no_grad():
                tVal.append(modelT(a).item())
                aVal.append(modelA(a).item())
                bVal.append(modelB(a).item())
                cVal.append(modelC(a).item())
                iBscaled = bScaler.transform(a.reshape(1,-1))
                iCscaled = cScaler.transform(a.reshape(1,-1))
                bStandardPred = modelBstandard(torch.from_numpy(iBscaled).float())
                cStandardPred = modelCstandard(torch.from_numpy(iCscaled).float())
                numpyBStandardPred = bStandardPred.detach().cpu().numpy()
                numpyCStandardPred = cStandardPred.detach().cpu().numpy()
                bInv = bTargetScaler.inverse_transform(numpyBStandardPred)
                cInv = cTargetScaler.inverse_transform(numpyCStandardPred)
                bStandardVal.append(bInv.item())
                cStandardVal.append(cInv.item())

        tS = 0
        aS = 0
        bS = 0
        cS = 0
        bStandardS = 0
        cStandardS = 0

        for i in tVal:
            tS = tS + i

        for i in aVal:
            aS = aS + i

        for i in bVal:
            bS = bS + i

        for i in cVal:
            cS = cS + i

        for i in bStandardVal:
            bStandardS = bStandardS + i

        for i in cStandardVal:
            cStandardS = cStandardS + i

        modelTSetVal.append (statistics.median(tVal))
        realTSetVal.append(T)
        modelASetVal.append(statistics.median(aVal))
        realASetVal.append(A)
        modelBSetVal.append(statistics.median(bVal))
        realBSetVal.append(B)
        modelCSetVal.append(statistics.median(cVal))
        realCSetVal.append(C)
        modelBstandardSetVal.append(statistics.median(bStandardVal))
        modelCstandardSetVal.append(statistics.median(cStandardVal))


modelTSetVal = list(map(float, modelTSetVal))
realTSetVal = list(map(float, realTSetVal))

modelASetVal = list(map(float, modelASetVal))
realASetVal = list(map(float, realASetVal))

modelBSetVal = list(map(float, modelBSetVal))
realBSetVal = list(map(float, realBSetVal))

modelCSetVal = list(map(float, modelCSetVal))
realCSetVal = list(map(float, realCSetVal))
modelBstandardSetVal = list(map(float, modelBstandardSetVal))
modelCstandardSetVal = list(map(float, modelCstandardSetVal))

r2A = stats.pearsonr(modelASetVal, realASetVal)
r2T = stats.pearsonr(modelTSetVal, realTSetVal)
r2B = stats.pearsonr(modelBSetVal, realBSetVal)
r2C = stats.pearsonr(modelCSetVal, realCSetVal)
r2Bstandard = stats.pearsonr(modelBstandardSetVal, realBSetVal)
r2Cstandard = stats.pearsonr(modelCstandardSetVal, realCSetVal)

print("R2 dla T: " + str((float(r2T[0])**2)))
print("R2 dla A: " + str((float(r2A[0])**2)))
print("R2 dla B: " + str((float(r2B[0])**2)))
print("R2 dla C: " + str((float(r2C[0])**2)))
print("R2 dla B standard: " + str((float(r2Bstandard[0])**2)))
print("R2 dla C standard: " + str((float(r2Cstandard[0])**2)))



