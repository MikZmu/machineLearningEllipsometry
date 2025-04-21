import joblib
import pandas as pd
from louis import backTranslate

import MLP_class
import predictSingle
import torch
import joblib
import getData_class
import get_Standarized_data
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

modelTT = ("modelT_48_32_16_8.pth")
modelAA = "modelA_48_32_16_8.pth"
modelBB = "modelB_256_256_256_128_64.pth"
modelCC = "modelC_64_32_32_16.pth"
modelBBstandard = "modelBstandard_48_32_32_16.pth"
modelCCstandard = "modelCstandard_48_32_32_16.pth"
modelT = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelTT))
modelA = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelAA))
modelB = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelBB))
modelC = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelCC))


modelBstandard = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelBBstandard))
modelCstandard = MLP_class.MLP(input_size=7, output_size=1, hidden_layers=predictSingle.decode_model(modelCCstandard))

bScaler = joblib.load("bScaler_featureScaler.pkl")
bTargetScaler = joblib.load("bScaler_targetScaler.pkl")
cTargetScaler = joblib.load("cScaler_targetScaler.pkl")
cScaler = joblib.load("cScaler_featureScaler.pkl")




if torch.cuda.is_available():
    modelT.load_state_dict(torch.load(modelTT))
else:
    modelT.load_state_dict(torch.load(modelTT, map_location=torch.device('cpu')))


if torch.cuda.is_available():
    modelA.load_state_dict(torch.load(modelAA))
else:
    modelA.load_state_dict(torch.load(modelAA, map_location=torch.device('cpu')))


if torch.cuda.is_available():
    modelC.load_state_dict(torch.load(modelCC))
    modelB.load_state_dict(torch.load(modelBB))
    modelBstandard.load_state_dict(torch.load(modelBBstandard))
    modelCstandard.load_state_dict(torch.load(modelCCstandard))
else:
    modelC.load_state_dict(torch.load(modelCC,  map_location=torch.device('cpu')))
    modelB.load_state_dict(torch.load(modelBB, map_location=torch.device('cpu')))
    modelBstandard.load_state_dict(torch.load(modelBBstandard, map_location=torch.device('cpu')))
    modelCstandard.load_state_dict(torch.load(modelCCstandard, map_location=torch.device('cpu')))


Tdata = getData_class.getData(['T'])
Adata = getData_class.getData(['A'])
Bdata = getData_class.getData(['B'])
Cdata = getData_class.getData(['C'])
BdataStandard = get_Standarized_data.get_Standarized_data('bScaler', ['B'])
CdataStandard = get_Standarized_data.get_Standarized_data('cScaler', ['C'])

tValues  = []
tTrain = []
aValues = []
aTrain = []
bValues = []
bTrain = []
cValues = []
cTrain = []
cStandardValues = []
csTrain = []
bStandardValues = []
bsTrain = []


for i in Tdata[2]:
    with torch.no_grad():
        tValues.append(modelT(i).item())

for i in Tdata[0]:
    with torch.no_grad():
        tTrain.append(modelT(i).item())


for i in Adata[2]:
    with torch.no_grad():
        aValues.append(modelA(i).item())

for i in Adata[0]:
    with torch.no_grad():
        aTrain.append(modelA(i).item())

for i in Bdata[2]:
    with torch.no_grad():
        bValues.append(modelB(i).item())

for i in Bdata[0]:
    with torch.no_grad():
        bTrain.append(modelB(i).item())

for i in Cdata[2]:
    with torch.no_grad():
        cValues.append(modelC(i).item())

for i in Cdata[0]:
    with torch.no_grad():
        cTrain.append(modelC(i).item())


featureNames = ['wavelength', 'psi65', 'del65', 'psi70', 'del70', 'psi75', 'del75']


for i in BdataStandard[2]:
    #a_df = pd.DataFrame(i.numpy().reshape(1, -1), columns=featureNames)
    #j = bScaler.transform(a_df)
    with torch.no_grad():
        bStandardPred = modelBstandard(i)
        #numpyBStandardPred = bStandardPred.detach().cpu().numpy()
        #bInv = bTargetScaler.inverse_transform(numpyBStandardPred)
        bStandardValues.append(bStandardPred.item())

for i in BdataStandard[0]:
    #a_df = pd.DataFrame(i.numpy().reshape(1, -1), columns=featureNames)
    #j = bScaler.transform(a_df)
    with torch.no_grad():
        bStandardPred = modelBstandard(i)
        #numpyBStandardPred = bStandardPred.detach().cpu().numpy()
        #bInv = bTargetScaler.inverse_transform(numpyBStandardPred)
        bsTrain.append(bStandardPred.item())



for i in CdataStandard[2]:
    #a_df = pd.DataFrame(i.numpy().reshape(1, -1), columns=featureNames)
    #j = cScaler.transform(a_df)
    with torch.no_grad():
        cStandardPred = modelCstandard(i)
        #numpyCStandardPred = cStandardPred.detach().cpu().numpy()
        #cInv = cTargetScaler.inverse_transform(numpyCStandardPred)
        cStandardValues.append(cStandardPred.item())

for i in CdataStandard[0]:
    #a_df = pd.DataFrame(i.numpy().reshape(1, -1), columns=featureNames)
    #j = cScaler.transform(a_df)
    with torch.no_grad():
        cStandardPred = modelCstandard(i)
        #numpyCStandardPred = cStandardPred.detach().cpu().numpy()
        #cInv = cTargetScaler.inverse_transform(numpyCStandardPred)
        csTrain.append(cStandardPred.item())

r2t = r2_score(Tdata[3].tolist(), tValues)
r2tTrain = r2_score(Tdata[1].tolist(), tTrain)
r2a = r2_score(Adata[3].tolist(), aValues)
r2aTrain = r2_score(Adata[1].tolist(), aTrain)
r2b = r2_score(Bdata[3].tolist(), bValues)
r2bTrain = r2_score(Bdata[1].tolist(), bTrain)
r2c = r2_score(Cdata[3].tolist(), cValues)
r2cTrain = r2_score(Cdata[1].tolist(), cTrain)
r2bs = r2_score(BdataStandard[3].tolist(), bStandardValues)
r2bsTrain = r2_score(BdataStandard[1].tolist(), bsTrain)
r2cs = r2_score(CdataStandard[3].tolist(), cStandardValues)
r2csTrain = r2_score(CdataStandard[1].tolist(), csTrain)

tlist = Tdata[3].tolist()
tlist1d = [item for sublist in tlist for item in sublist]
ttlist = Tdata[1].tolist()
ttlist1d = [item for sublist in ttlist for item in sublist]

alist = Adata[3].tolist()
alist1d = [item for sublist in alist for item in sublist]
atlist = Adata[1].tolist()
atlist1d = [item for sublist in atlist for item in sublist]

blist = Bdata[3].tolist()
blist1d = [item for sublist in blist for item in sublist]
btlist = Bdata[1].tolist()
btlist1d = [item for sublist in btlist for item in sublist]

clist = Cdata[3].tolist()
clist1d = [item for sublist in clist for item in sublist]
ctlist = Cdata[1].tolist()
ctlist1d = [item for sublist in ctlist for item in sublist]


blistStandard = BdataStandard[3].tolist()
blistStandard1d = [item for sublist in blistStandard for item in sublist]
btlistStandard = BdataStandard[1].tolist()
btlistStandard1d = [item for sublist in btlistStandard for item in sublist]

clistStandard = CdataStandard[3].tolist()
clistStandard1d = [item for sublist in clistStandard for item in sublist]
ctlistStandard = CdataStandard[1].tolist()
ctlistStandard1d = [item for sublist in ctlistStandard for item in sublist]



pt = pearsonr(tlist1d, tValues)
ptt = pearsonr(ttlist1d, tTrain)

pa = pearsonr(alist1d, aValues)
pat = pearsonr(atlist1d, aTrain)

pb = pearsonr(blist1d, bValues)
pbt = pearsonr(btlist1d, bTrain)

pc = pearsonr(clist1d, cValues)
pct = pearsonr(ctlist1d, cTrain)

pbs = pearsonr(blistStandard1d , bStandardValues)
pbst = pearsonr(btlistStandard1d , bsTrain)

pcs = pearsonr(clistStandard1d, cStandardValues)
pcst = pearsonr(ctlistStandard1d, csTrain)

print("R2 T: ", pt[0]**2)
print("R2 T Train: ", ptt[0]**2)
print("R2 A: ", pa[0]**2)
print("R2 A Train: ", pat[0]**2)
print("R2 B: ", pb[0]**2)
print("R2 B Train: ", pbt[0]**2)
print("R2 C: ", pc[0]**2)
print("R2 C Train: ", pct[0]**2)
print("R2 B Standard: ", pbs[0]**2)
print("R2 B Standard Train: ", pbst[0]**2)
print("R2 C Standard: ", pcs[0]**2)
print("R2 C Standard Train: ", pcst[0]**2)
