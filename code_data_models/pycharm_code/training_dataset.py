import os
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import MLP_class
from training_sample import training_sample
import training_class


class training_dataset:
    def __init__(self, dataset_folder):
        self.samples = self.gather_samples(dataset_folder)





    def gather_samples(self, dataset_folder):
        all_items = os.listdir(dataset_folder)
        files = [item for item in all_items if os.path.isfile(os.path.join(dataset_folder, item))]
        samples = []
        for i in files:
            sample = training_sample(os.path.join(dataset_folder, i))
            samples.append(sample)

        return samples

    def return_as_tensors_split(self, feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T']):
        df = pd.DataFrame()
        for sample in self.samples:
            df.concat(sample.data[feature_columns], ignore_index=True)
        features = df[feature_columns]
        targets = df[target_columns]
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        x_train = torch.from_numpy(x_train.to_numpy(dtype=np.float32))
        x_test = torch.from_numpy(x_test.to_numpy(dtype=np.float32))
        y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
        y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))
        return [x_train, y_train, x_test, y_test]

    def return_as_tensors(self, columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T']):
        df = pd.DataFrame()

        for sample in self.samples:
            df = pd.concat([df, sample.data[columns + target_columns]], ignore_index=True)

        df.columns = columns + target_columns
        features = df[columns]
        targets = df[target_columns]
        features = torch.from_numpy(features.to_numpy(dtype=np.float32))
        targets = torch.from_numpy(targets.to_numpy(dtype=np.float32))
        return [features, targets]


    def get_total_r2_score(self, model, features = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], targets = ['T']):
        model = MLP_class.MLP.create_and_load(model, input_size=len(features), output_size=len(targets))
        model.eval()
        data = self.return_as_tensors(features, targets)
        features = data[0]
        targets = data[1]
        with torch.no_grad():
            predictions = model(features)
            predictions = predictions.flatten().tolist()

        pearson = pearsonr(predictions, targets.flatten().tolist())
        r2_score = pearson[0] ** 2
        return r2_score

    def return_as_flat_tensors(self, feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T']):

        targetDf = pd.DataFrame()
        featureDf = pd.DataFrame()

        for sample in self.samples:
            features, targets = sample.return_as_flat_df(feature_columns, target_columns)
            featureDf = pd.concat([featureDf, features], ignore_index=True)
            targetDf = pd.concat([targetDf, targets], ignore_index=True)
            

        x_train, x_test, y_train, y_test = train_test_split(featureDf, targetDf, test_size=0.2, random_state=42)
        x_train = torch.from_numpy(x_train.to_numpy(dtype=np.float32))
        print(x_train[0].shape)
        x_test = torch.from_numpy(x_test.to_numpy(dtype=np.float32))
        y_train = torch.from_numpy(y_train.to_numpy(dtype=np.float32))
        y_test = torch.from_numpy(y_test.to_numpy(dtype=np.float32))

        return [x_train, x_test, y_train, y_test]


    def for_sample(self):
        for sample in self.samples:
            sample.return_as_flat_tensors()


    def train_flattened(self,model_name = "default", feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T'], hidden_layers = [256, 128, 64, 32], loss = nn.MSELoss(), save_folder = "models"):

        data = self.return_as_flat_tensors(feature_columns, target_columns)

        if model_name == "default":
            code_layers = ""
            for layer in hidden_layers:
                code_layers += str(layer) + "_"
            code_layers = code_layers[:-1]
            model_name = "model" + str(target_columns) + "_" + str(code_layers) + ".pth"

        model = MLP_class.MLP(input_size=497, output_size=4, hidden_layers=hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        save_path = os.path.join(save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        training_class.train_model(model, loss, optimizer, data[0], data[2], data[1], data[3], save_path=save_path, batch_size=0)


    def train(self, model_name = "default", feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T'], hidden_layers = [64, 32, 32, 16], loss = nn.MSELoss(), save_folder = "models"):

        data = self.return_as_tensors_split(feature_columns, target_columns)

        if model_name == "default":
            code_layers = ""
            for layer in hidden_layers:
                code_layers += str(layer) + "_"
            code_layers = code_layers[:-1]
            model_name = "model" + str(target_columns) + "_" + str(code_layers) + ".pth"

        model = MLP_class.MLP(input_size=497, output_size=4, hidden_layers=hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        save_path = os.path.join(save_folder, model_name)
        os.makedirs(save_folder, exist_ok=True)
        training_class.train_model(model, loss, optimizer, data[0], data[2], data[1], data[3], save_path=save_path, batch_size=0)







folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(folder)
folder_path = os.path.join(parent_folder,"datasets", "new_Si_jaw_delta", "")
dataset = training_dataset(folder_path)
training_dataset.get_total_r2_score()
