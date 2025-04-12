import os
import pandas as pd
import torch
import MLP_class
from scipy import stats


modelT = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])
modelT.load_state_dict(torch.load("modelT.pth"))
modelA = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=[128, 64, 64, 32])
modelA.load_state_dict(torch.load("modelA.pth"))
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

modelSetVal = []
realSetVal = []
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
                aVal.append(modelA.)


        tS = 0

        for i in values:
            tS = tS + i

        modelSetVal.append (tS / len(values))
        realSetVal.append(T)

modelSetVal = list(map(float, modelSetVal))
realSetVal = list(map(float, realSetVal))

R2 = stats.pearsonr(modelSetVal, realSetVal)
print(float(R2[0])*float(R2[0]))