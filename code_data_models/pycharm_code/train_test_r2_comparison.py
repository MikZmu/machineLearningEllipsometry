import os

import torch

import getData_class

import MLP_class

from predictSingle import decode_model

from scipy.stats import pearsonr

def create_and_load(model_folder,model_name, input_size, output_size):
    model_path = os.path.join(model_folder, model_name)
    model = MLP_class.MLP(input_size, output_size, hidden_layers=decode_model(model_name))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def list_to_float(lst):
    return [float(i) for i in lst]

folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(folder)
print(parent_folder)
folder_path = os.path.join(parent_folder,"datasets", "new_Si_jaw_delta", "")
model_folder = os.path.join(parent_folder,"pycharm_code" ,"")
half_data_folder = os.path.join(parent_folder,"datasets", "half_new_Si_jaw_delta", "")
quarter_data_folder = os.path.join(parent_folder,"datasets", "quarter_new_Si_jaw_delta", "")




Adata = getData_class.getData(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["A"], folder_path)
Bdata = getData_class.getData(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["B"], folder_path)
Cdata = getData_class.getData(["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["C"], folder_path)
Csdata = getData_class.get_Standarized_data("cScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["C"], folder_path)
Bsdata = getData_class.get_Standarized_data("bScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["B"], folder_path)








def calculate_set_r2(model, x,y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = pred.flatten().tolist()
        y= y.flatten().tolist()
    return (pearsonr(pred, y)[0]**2)

def calculate_r2(model, data):
    model.eval()
    with torch.no_grad():
        predTrain = model(data[0])
        predTrain = predTrain.flatten().tolist()
        predTest = model(data[2])
        predTest = predTest.flatten().tolist()
        yTrain = data[1].flatten().tolist()
        yTest = data[3].flatten().tolist()
    return ([pearsonr(predTrain, yTrain)[0]**2, pearsonr(predTest, yTest)[0]**2])

A_32_16_16_8 = create_and_load(model_folder, "modelA_32_16_16_8.pth", 7, 1)
print(f"modelA_32_16_16_8 R2 Train: {calculate_r2(A_32_16_16_8, Adata)[0]} Test: {calculate_r2(A_32_16_16_8, Adata)[1]}")
A_48_32_24_12 = create_and_load(model_folder, "modelA_48_32_24_12.pth", 7, 1)
print(f"modelA_48_32_24_12 R2 Train: {calculate_r2(A_48_32_24_12, Adata)[0]} Test: {calculate_r2(A_48_32_24_12, Adata)[1]}")
A_64_32_32_16 = create_and_load(model_folder, "modelA_64_32_32_16.pth", 7, 1)
print(f"modelA_64_32_32_16 R2 Train: {calculate_r2(A_64_32_32_16, Adata)[0]} Test: {calculate_r2(A_64_32_32_16, Adata)[1]}")
B_32_16_16_8 = create_and_load(model_folder, "modelB_32_16_16_8.pth", 7, 1)
print(f"modelB_32_16_16_8 R2 Train: {calculate_r2(B_32_16_16_8, Bdata)[0]} Test: {calculate_r2(B_32_16_16_8, Bdata)[1]}")
B_48_32_24_12 = create_and_load(model_folder, "modelB_48_32_24_12.pth", 7, 1)
print(f"modelB_48_32_24_12 R2 Train: {calculate_r2(B_48_32_24_12, Bdata)[0]} Test: {calculate_r2(B_48_32_24_12, Bdata)[1]}")
B_64_32_32_16 = create_and_load(model_folder, "modelB_64_32_32_16.pth", 7, 1)
print(f"modelB_64_32_32_16 R2 Train: {calculate_r2(B_64_32_32_16, Bdata)[0]} Test: {calculate_r2(B_64_32_32_16, Bdata)[1]}")
C_32_16_16_8 = create_and_load(model_folder, "modelC_32_16_16_8.pth", 7, 1)
print(f"modelC_32_16_16_8 R2 Train: {calculate_r2(C_32_16_16_8, Cdata)[0]} Test: {calculate_r2(C_32_16_16_8, Cdata)[1]}")
C_48_32_24_12 = create_and_load(model_folder, "modelC_48_32_24_12.pth", 7, 1)
print(f"modelC_48_32_24_12 R2 Train: {calculate_r2(C_48_32_24_12, Cdata)[0]} Test: {calculate_r2(C_48_32_24_12, Cdata)[1]}")
C_64_32_32_16 = create_and_load(model_folder, "modelC_64_32_32_16.pth", 7, 1)
print(f"modelC_64_32_32_16 R2 Train: {calculate_r2(C_64_32_32_16, Cdata)[0]} Test: {calculate_r2(C_64_32_32_16, Cdata)[1]}")

Cst_32_16_16_8 = create_and_load(model_folder, "modelCstandard_32_16_16_8.pth", 7, 1)
print(f"modelCstandard_32_16_16_8 R2 Train: {calculate_r2(Cst_32_16_16_8, Csdata)[0]} Test: {calculate_r2(Cst_32_16_16_8, Csdata)[1]}")
Cst_48_32_24_12 = create_and_load(model_folder, "modelCstandard_48_32_24_12.pth", 7, 1)
print(f"modelCstandard_48_32_24_12 R2 Train: {calculate_r2(Cst_48_32_24_12, Csdata)[0]} Test: {calculate_r2(Cst_48_32_24_12, Csdata)[1]}")
Cst_64_32_32_16 = create_and_load(model_folder, "modelCstandard_64_32_32_16.pth", 7, 1)
print(f"modelCstandard_64_32_32_16 R2 Train: {calculate_r2(Cst_64_32_32_16, Csdata)[0]} Test: {calculate_r2(Cst_64_32_32_16, Csdata)[1]}")
Bst_32_16_16_8 = create_and_load(model_folder, "modelBstandard_32_16_16_8.pth", 7, 1)
print(f"modelBstandard_32_16_16_8 R2 Train: {calculate_r2(Bst_32_16_16_8, Bsdata)[0]} Test: {calculate_r2(Bst_32_16_16_8, Bsdata)[1]}")
Bst_48_32_24_12 = create_and_load(model_folder, "modelBstandard_48_32_24_12.pth", 7, 1)
print(f"modelBstandard_48_32_24_12 R2 Train: {calculate_r2(Bst_48_32_24_12, Bsdata)[0]} Test: {calculate_r2(Bst_48_32_24_12, Bsdata)[1]}")
Bst_64_32_32_16 = create_and_load(model_folder, "modelBstandard_64_32_32_16.pth", 7, 1)
print(f"modelBstandard_64_32_32_16 R2 Train: {calculate_r2(Bst_64_32_32_16, Bsdata)[0]} Test: {calculate_r2(Bst_64_32_32_16, Bsdata)[1]}")

csXdatacombined = torch.cat((Csdata[0], Csdata[2]), dim=0)
csYdatacombined = torch.cat((Csdata[1], Csdata[3]), dim=0)
bsXdatacombined = torch.cat((Bsdata[0], Bsdata[2]), dim=0)
bsYdatacombined = torch.cat((Bsdata[1], Bsdata[3]), dim=0)

print(f"combined modelCstandard_64_32_32_16 R2: {calculate_set_r2(Cst_64_32_32_16, csXdatacombined, csYdatacombined)}")
print(f"combined modelBstandard_64_32_32_16 R2: {calculate_set_r2(Bst_64_32_32_16, bsXdatacombined, bsYdatacombined)}")
print(f"combined modelCstandard_48_32_24_12 R2: {calculate_set_r2(Cst_48_32_24_12, csXdatacombined, csYdatacombined)}")
print(f"combined modelBstandard_48_32_24_12 R2: {calculate_set_r2(Bst_48_32_24_12, bsXdatacombined, bsYdatacombined)}")
print(f"combined modelCstandard_32_16_16_8 R2: {calculate_set_r2(Cst_32_16_16_8, csXdatacombined, csYdatacombined)}")
print(f"combined modelBstandard_32_16_16_8 R2: {calculate_set_r2(Bst_32_16_16_8, bsXdatacombined, bsYdatacombined)}")



cshalfdata = getData_class.get_Standarized_data("cScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["C"], half_data_folder)
bshalfdata = getData_class.get_Standarized_data("bScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["B"], half_data_folder)


c_half_64_32_32_16 = create_and_load(model_folder, "modelhalfCstandard_64_32_32_16.pth", 7, 1)
print(f"modelhalfCstandard_64_32_32_16 R2 Train: {calculate_r2(c_half_64_32_32_16, cshalfdata)[0]} Test: {calculate_r2(c_half_64_32_32_16, cshalfdata)[1]}")
b_half_64_32_32_16 = create_and_load(model_folder, "modelhalfBstandard_64_32_32_16.pth", 7, 1)
print(f"modelhalfBstandard_64_32_32_16 R2 Train: {calculate_r2(b_half_64_32_32_16, bshalfdata)[0]} Test: {calculate_r2(b_half_64_32_32_16, bshalfdata)[1]}")

cs_halfX_data_combined = torch.cat((cshalfdata[0], cshalfdata[2]), dim=0)
cs_halfY_data_combined = torch.cat((cshalfdata[1], cshalfdata[3]), dim=0)
bs_halfX_data_combined = torch.cat((bshalfdata[0], bshalfdata[2]), dim=0)
bs_halfY_data_combined = torch.cat((bshalfdata[1], bshalfdata[3]), dim=0)

print(f"combined modelhalfCstandard_64_32_32_16 R2: {calculate_set_r2(c_half_64_32_32_16, cs_halfX_data_combined, cs_halfY_data_combined)}")
print(f"combined modelhalfBstandard_64_32_32_16 R2: {calculate_set_r2(b_half_64_32_32_16, bs_halfX_data_combined, bs_halfY_data_combined)}")



cquarterdata = getData_class.get_Standarized_data("cScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["C"], quarter_data_folder)
bquarterdata = getData_class.get_Standarized_data("bScaler",["wavelength","psi65", "del65", "psi70", "del70", "psi75", "del75"], ["B"], quarter_data_folder)


c_quarter_64_32_32_16 = create_and_load(model_folder, "modelquarterCstandard_64_32_32_16.pth", 7, 1)
print(f"modelquarterCstandard_64_32_32_16 R2 Train: {calculate_r2(c_quarter_64_32_32_16, cquarterdata)[0]} Test: {calculate_r2(c_quarter_64_32_32_16, cquarterdata)[1]}")

b_quarter_64_32_32_16 = create_and_load(model_folder, "modelquarterBstandard_64_32_32_16.pth", 7, 1)
print(f"modelquarterBstandard_64_32_32_16 R2 Train: {calculate_r2(b_quarter_64_32_32_16, bquarterdata)[0]} Test: {calculate_r2(b_quarter_64_32_32_16, bquarterdata)[1]}")


cs_halfX_data_combined = torch.cat((cquarterdata[0], cquarterdata[2]), dim=0)
cs_halfY_data_combined = torch.cat((cquarterdata[1], cquarterdata[3]), dim=0)
bs_halfX_data_combined = torch.cat((bquarterdata[0], bquarterdata[2]), dim=0)
bs_halfY_data_combined = torch.cat((bquarterdata[1], bquarterdata[3]), dim=0)
print(f"combined modelquarterCstandard_64_32_32_16 R2: {calculate_set_r2(c_quarter_64_32_32_16, cs_halfX_data_combined, cs_halfY_data_combined)}")
print(f"combined modelquarterBstandard_64_32_32_16 R2: {calculate_set_r2(b_quarter_64_32_32_16, bs_halfX_data_combined, bs_halfY_data_combined)}")





