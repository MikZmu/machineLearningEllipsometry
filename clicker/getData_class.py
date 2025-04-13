import torch
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def getData(target_columns=['T', 'A', 'B', 'C']):
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

        if ("-" in C):
            C = C.removesuffix("e-")
            C = float(C) * 10 ** -5
        elif ("e" in C):
            C = C.removesuffix("e")
            C = float(C) * 10 ** -5

        if (float(C) > 1):
            C = float(C) * 10 ** -5

        dataHelper['C'] = C
        dataFrame = pd.concat([dataFrame, dataHelper], ignore_index=True)

    x_train, x_test, y_train, y_test = train_test_split(
        dataFrame[['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']],
        dataFrame[target_columns],
        test_size=0.2,
        random_state=42
    )
    x_train = torch.from_numpy(x_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
    y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))
    return [x_train, y_train, x_test, y_test]

x = getData('T')

print(x[1])