import os
import pickle

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.preprocessing import StandardScaler
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


    def return_as_flat_standardized_tensors(self, feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T'], scalerName = "standardScaler"):
        targetDf = pd.DataFrame()
        featureDf = pd.DataFrame()

        for sample in self.samples:
            features, targets = sample.return_as_flat_df(feature_columns, target_columns)
            featureDf = pd.concat([featureDf, features], ignore_index=True)
            targetDf = pd.concat([targetDf, targets], ignore_index=True)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()



        x_scaled = x_scaler.fit_transform(featureDf)
        y_scaled = y_scaler.fit_transform(targetDf)


        with open(scalerName + "Y" + ".pkl", "wb") as f:
            pickle.dump(y_scaler, f)

        with open(scalerName + "X" + ".pkl", "wb") as f:
            pickle.dump(x_scaler, f)



        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
        x_train = torch.from_numpy(x_train.astype(np.float32))
        x_test = torch.from_numpy(x_test.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        return [x_train, x_test, y_train, y_test]




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


    def train_standardized_flattened(self,model_name = "defaultStandarized", feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T'], hidden_layers = [512, 256, 128, 64], loss = nn.MSELoss(), save_folder = "models"):
        data = self.return_as_flat_standardized_tensors(feature_columns, target_columns)

        if model_name == "defaultStandarized":
            code_layers = ""
            for layer in hidden_layers:
                code_layers += str(layer) + "_"
            code_layers = code_layers[:-1]
            model_name = "modelStanderd" + str(target_columns) + "_" + str(code_layers) + ".pth"

        model = MLP_class.MLP(input_size=497, output_size=1, hidden_layers=hidden_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.001)
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


    def test_r2_flattened(self, x_Scaler_name, y_Scaler_name, model_name = "default", feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['T', 'A', 'B', 'C']):

        model = MLP_class.MLP.create_and_load(model_name, input_size=497, output_size=1)
        model.eval()
        # with open(x_Scaler_name + ".pkl", "rb") as f:
        #     x_scaler = pickle.load(f)
        #
        # with open(y_Scaler_name + ".pkl", "rb") as f:
        #     y_scaler = pickle.load(f)

        data = self.return_as_flat_standardized_tensors(feature_columns, target_columns)
        features_tr = data[0]
        features_te = data[1]
        targets_tr = data[2]
        targets_te = data[3]

        with torch.no_grad():
            pred_tr = model(features_tr)
            pred_te = model(features_te)

        pred_tr = [pred_tr[:,i] for i in range(pred_tr.size(1))]
        pred_te = [pred_te[:,i] for i in range(pred_te.size(1))]
        targets_tr = [targets_tr[:,i] for i in range(targets_tr.size(1))]
        targets_te = [targets_te[:,i] for i in range(targets_te.size(1))]
        T_pred_tr = pred_tr[0].tolist()
        # A_pred_tr = pred_tr[1].tolist()
        # B_pred_tr = pred_tr[2].tolist()
        # C_pred_tr = pred_tr[3].tolist()

        T_pred_te = pred_te[0].tolist()
        # A_pred_te = pred_te[1].tolist()
        # B_pred_te = pred_te[2].tolist()
        # C_pred_te = pred_te[3].tolist()

        T_target_tr = targets_tr[0].tolist()
        # A_target_tr = targets_tr[1].tolist()
        # B_target_tr = targets_tr[2].tolist()
        # C_target_tr = targets_tr[3].tolist()

        T_target_te = targets_te[0].tolist()
        # A_target_te = targets_te[1].tolist()
        # B_target_te = targets_te[2].tolist()
        # C_target_te = targets_te[3].tolist()

        T_tr_r2 = pearsonr(T_pred_tr, T_target_tr)[0] ** 2
        # A_tr_r2 = pearsonr(A_pred_tr, A_target_tr)[0] ** 2
        # B_tr_r2 = pearsonr(B_pred_tr, B_target_tr)[0] ** 2
        # C_tr_r2 = pearsonr(C_pred_tr, C_target_tr)[0] ** 2

        T_te_r2 = pearsonr(T_pred_te, T_target_te)[0] ** 2
        # A_te_r2 = pearsonr(A_pred_te, A_target_te)[0] ** 2
        # B_te_r2 = pearsonr(B_pred_te, B_target_te)[0] ** 2
        # C_te_r2 = pearsonr(C_pred_te, C_target_te)[0] ** 2

        print("T Train R2: ", T_tr_r2)
        # print("A Train R2: ", A_tr_r2)
        # print("B Train R2: ", B_tr_r2)
        # print("C Train R2: ", C_tr_r2)

        print("T Test R2: ", T_te_r2)
        # print("A Test R2: ", A_te_r2)
        # print("B Test R2: ", B_te_r2)
        # print("C Test R2: ", C_te_r2)










folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(folder)
folder_path = os.path.join(parent_folder,"datasets", "new_Si_jaw_delta", "")
dataset = training_dataset(folder_path)
dataset.train_standardized_flattened(feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'], target_columns = ['C'], hidden_layers = [5,5], loss = nn.MSELoss(), save_folder = "models")
# dataset.test_r2_flattened("standardScalerX",
#                           "standardScalerY",
#                           model_name = "models//modelStanderd['C']_96_48_24.pth",
#                           feature_columns=['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75'],
#                           target_columns = ['C'])
