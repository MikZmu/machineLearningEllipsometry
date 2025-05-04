import os
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_sample_image
import MLP_class


class training_sample:

    def __init__(self, file_path):
        self.T = self.decode_sample(file_path)[0]
        self.A = self.decode_sample(file_path)[1]
        self.B = self.decode_sample(file_path)[2]
        self.C = self.decode_sample(file_path)[3]
        self.data = self.load_data(file_path, self.T, self.A, self.B, self.C)


    def decode_sample(self, filename):
        filename = os.path.basename(filename)
        info = filename.split("_")
        T = info[0]
        A = info[1]
        B = info[2]
        C = info[3]
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

    def return_as_flat_df(self, feature_columns = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T', 'A', 'B', 'C']):
        features = self.data[feature_columns]
        targets = self.data[target_columns]
        targets = targets.iloc[:1]
        features = features.values.reshape(1, -1)
        targets = targets.values.reshape(1, -1)
        features = pd.DataFrame(features)
        targets = pd.DataFrame(targets)
        return features, targets

    def return_as_flat_tensors(self,feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'],target_columns=['T', 'A', 'B', 'C']):
        features = self.data[feature_columns]
        targets = self.data[target_columns]
        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))
        features = features.reshape(1, -1)
        targets = targets[:1]
        targets = targets.reshape(1, -1)

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

    def get_sample_info(self):
        return self.T, self.A, self.B, self.C

    def print_sample_info(self):
        print(f'T: {self.T}, A: {self.A}, B: {self.B}, C: {self.C}')


    def predict_median(self, model, features, output_size):
        model = MLP_class.MLP.create_and_load(model, features.shape[1], output_size)
        model.eval()
        features_as_tensors = self.features_as_tensors(features)
        with torch.no_grad():
            predictions = model(features_as_tensors)
        predictions = predictions.flatten().tolist()
        median = np.median(predictions)
        return median
  

    def get_rmse_mean(self, data):
        data = np.sqrt(np.mean((data - data.mean()) ** 2))
        return data

    def get_rmse_median(self, data):
        data = np.sqrt(np.mean((data - data.median()) ** 2))
        return data

    def return_flattened(self):
        data = self.data
        data.drop("T", axis=1, inplace=True)
        data.drop("A", axis=1, inplace=True)
        data.drop("B", axis=1, inplace=True)
        data.drop("C", axis=1, inplace=True)
        data.values.flatten()
        return data





