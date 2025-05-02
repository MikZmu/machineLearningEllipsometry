import os

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_sample_image

from code_data_models.pycharm_code import MLP_class


class training_sample:

    def __init__(self, file_path):
        self.T = self.decode_sample(file_path)[0]
        self.A = self.decode_sample(file_path)[1]
        self.B = self.decode_sample(file_path)[2]
        self.C = self.decode_sample(file_path)[3]
        self.data = self.load_data(file_path, self.T, self.A, self.B, self.C)


    def decode_sample(self, filename):
        filename = os.path.basename(filename)
        print(filename)
        filename.split("_")
        T = filename[0]
        A = filename[1]
        B = filename[2]
        C = filename[3]
        C = C.removesuffix(".txt")
        return [T, A, B, C]


    def load_data(self, filename, T, A, B, C):
        dataHelper = pd.read_csv(filename, sep='\t', header=None, index_col=False)
        dataHelper = dataHelper.drop(index=[0])
        dataHelper = dataHelper.drop(columns=[7])
        dataHelper.columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']
        dataHelper['T'] = T
        dataHelper['A'] = A
        dataHelper['B'] = B
        dataHelper['C'] = C
        return dataHelper

    def return_as_2dlist(self, data, feature_columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T', 'A', 'B', 'C']):
        features = data[feature_columns]
        targets = data[target_columns]
        return features, targets

    def return_as_tensors(self, feature_columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T', 'A', 'B', 'C']):
        features = self.data=[feature_columns]
        targets = self.data[target_columns]
        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))
        return features, targets

    def features_as_tensors(self, feature_columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']):
        features = self.data[feature_columns]
        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        return features

    def predict_mean(self, model, features, output_size):
        model = MLP_class.MLP.create_and_load(model,features.shape[1], output_size)
        model.eval()
        features_as_tensors = self.features_as_tensors(features)
        with torch.no_grad():
            predictions = model(features_as_tensors)
        predictions = predictions.flatten().tolist()
        mean = np.mean(predictions)
        return mean

    def return_standarized(self, scaler, columns):
        data = self.data[columns]
        data = scaler.transform(data)
        return data

    """def predict_median(self):
        data = data.median(axis=0)
        return data
  
    def predict_rows(self):
        data = data.iloc[0]
        return data

    def get_rmse_mean(self, data):
        data = np.sqrt(np.mean((data - data.mean()) ** 2))
        return data

    def get_rmse_median(self, data):
        data = np.sqrt(np.mean((data - data.median()) ** 2))
        return data"""
