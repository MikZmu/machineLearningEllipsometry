import os
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from code_data_models.pycharm_code import MLP_class
from training_sample import training_sample


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

        print(df.head())
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



folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(folder)
folder_path = os.path.join(parent_folder,"datasets", "new_Si_jaw_delta", "")

dataset = training_dataset(folder_path)
print(dataset.get_total_r2_score("modelA_64_32_32_16.pth", ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], ['A']))