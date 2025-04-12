import os
import pandas as pd
import torch
import MLP_class
from scipy import stats
from sklearn.metrics import mean_squared_error



modelT = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])
if(torch.cuda.is_available()):
    modelT.load_state_dict(torch.load("modelT.pth"))
else:
    modelT.load_state_dict(torch.load("modelT.pth", map_location=torch.device('cpu')))


modelA = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])
if(torch.cuda.is_available()):
    modelA.load_state_dict(torch.load("modelA.pth"))
else:
    modelA.load_state_dict(torch.load("modelA.pth", map_location=torch.device('cpu')))

modelB = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])

modelC = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])
if(torch.cuda.is_available()):
    #modelC.load_state_dict(torch.load("modelC.pth"))
    modelB.load_state_dict(torch.load("modelB.pth"))
else:
    #modelC.load_state_dict(torch.load("modelC.pth", map_location=torch.device('cpu')))
    modelB.load_state_dict(torch.load("modelB.pth", map_location=torch.device('cpu')))



project_folder = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(project_folder,"datasets", "Si_jaw_delta", "")
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

for i in files:
        print(i)
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

        values = [[tVal,aVal,bVal,cVal]]

        for i in x:
            with torch.no_grad():
                tVal.append(modelT(i).item())
                aVal.append(modelA(i).item())
                bVal.append(modelB(i).item())


        tS = 0
        aS = 0
        bS = 0
        cS = 0

        for i in tVal:
            tS = tS + i

        for i in aVal:
            aS = aS + i

        for i in bVal:
            bS = bS + i

        modelTSetVal.append (tS / len(values))
        realTSetVal.append(T)
        modelASetVal.append(aS / len(values))
        realASetVal.append(A)
        modelBSetVal.append(bS / len(values))
        realBSetVal.append(B)

modelTSetVal = list(map(float, modelTSetVal))
realTSetVal = list(map(float, realTSetVal))

modelASetVal = list(map(float, modelASetVal))
realASetVal = list(map(float, realASetVal))

modelBSetVal = list(map(float, modelBSetVal))
realBSetVal = list(map(float, realBSetVal))

r2A = stats.pearsonr(modelASetVal, realASetVal)
r2T = stats.pearsonr(modelTSetVal, realTSetVal)
r2B = stats.pearsonr(modelBSetVal, realBSetVal)

print("R2 dla T: " + str((float(r2T[0])**2)))
print("R2 dla A: " + str((float(r2A[0])**2)))
print("R2 dla B: " + str((float(r2B[0])**2)))
print(mean_squared_error(modelBSetVal, realBSetVal))


