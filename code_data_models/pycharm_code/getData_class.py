import joblib
import torch
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# This function loads data from a specified folder, processes it, and splits it into training and testing sets. Returns the processed data as PyTorch tensors.
# Keeping random_state fixed to ensure reproducibility as long as the data is not changed.

def getData(feature_columns ,target_columns, folder):


    all_items = os.listdir(folder)

    files = [item for item in all_items if os.path.isfile(os.path.join(folder, item))]

    dataFrame = pd.DataFrame()


    for i in files:
        dataHelper = pd.read_csv(folder + i, sep='\t', header=None, index_col=False)
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
        dataHelper['C'] = C
        dataFrame = pd.concat([dataFrame, dataHelper], ignore_index=True)

    print(dataFrame.columns)
    x_train, x_test, y_train, y_test = train_test_split(
        dataFrame[feature_columns],
        dataFrame[target_columns],
        test_size=0.2,
        random_state=42
    )
    x_train = torch.from_numpy(x_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
    y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))
    return [x_train, y_train, x_test, y_test]

# This function standardizes the data using StandardScaler, saves the scalers, and splits the data into training and testing sets. Returns the processed data as PyTorch tensors.
# It also handles the conversion of certain values in the dataset based on specific conditions.
# The function takes a scaler name, target columns, and an optional folder path as input parameters.
#Keeping random_state fixed to ensure reproducibility as long as the data is not changed.

def get_Standarized_data(scalerName,feature_columns ,target_columns,folder):

    all_items = os.listdir(folder)

    files = [item for item in all_items if os.path.isfile(os.path.join(folder, item))]

    dataFrame = pd.DataFrame()

    for i in files:
        dataHelper = pd.read_csv(folder + i, sep='\t', header=None, index_col=False)
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

        dataHelper['C'] = C
        dataFrame = pd.concat([dataFrame, dataHelper], ignore_index=True)

    #print(dataFrame.head())

    features = dataFrame[feature_columns]
    targets = dataFrame[target_columns]

    featureScaler = StandardScaler()

    standarized_features = featureScaler.fit_transform(features)

    targetScaler = StandardScaler()

    standarized_targets = targetScaler.fit_transform(targets)

    joblib.dump(featureScaler, scalerName + '_featureScaler.pkl')
    joblib.dump(targetScaler, scalerName + '_targetScaler.pkl')

    x_train, x_test, y_train, y_test = train_test_split(
        standarized_features,
        standarized_targets,
        test_size=0.2,
        random_state=42
    )
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    return [x_train, y_train, x_test, y_test]


def get_data_chunks(feature_columns ,target_columns, folder):


    all_items = os.listdir(folder)

    files = [item for item in all_items if os.path.isfile(os.path.join(folder, item))]

    dfList = []

    for i in files:
        dataHelper = pd.read_csv(folder + i, sep='\t', header=None, index_col=False)
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
        dataHelper['C'] = C
        features = dataHelper[feature_columns]
        targets = dataHelper[target_columns]

        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))

        dfList.append([features, targets])

    return dfList


def get_standarized_chunks(feature_scaler, target_scaler, feature_columns, target_columns, folder):

    all_items = os.listdir(folder)
    files = [item for item in all_items if os.path.isfile(os.path.join(folder, item))]

    dfList = []

    for i in files:
        dataHelper = pd.read_csv(folder + i, sep='\t', header=None, index_col=False)
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
        dataHelper['C'] = C
        features = dataHelper[feature_columns]
        targets = dataHelper[target_columns]

        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))

        features_scaled = feature_scaler.transform(features)
        targets_scaled = target_scaler.transform(targets)

        dfList.append([features_scaled, targets_scaled])

    return dfList

def get_conv_data(feature_columns, target_columns, folder):
    all_items = os.listdir(folder)
    files = [item for item in all_items if os.path.isfile(os.path.join(folder, item))]
    dfList = []
    for i in files:
        dataHelper = pd.read_csv(folder + i, sep='\t', header=None, index_col=False)
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
        dataHelper['C'] = C
        features = dataHelper[feature_columns]
        targets = dataHelper[target_columns]

        # Convert to tensors
        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))

        # Reshape features to (batch_size, input_channels, height, width)
        features = features.view(-1, 1, 71, 7)  # Adjust height and width as needed
        targets = targets.unsqueeze(1)  # Add channel dimension to targets

        print(features.shape)  # Debugging: Check the shape
        dfList.append([features, targets])

    return dfList


