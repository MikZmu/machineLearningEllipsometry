scalers:
    contains files that are used to scale the data

pycham_code:
    contains files that are used to run the code in PyCharm. for now model training is performed using pycharm due to long-term instability of the jupyter notebook

models:
    contains models trained. model names contain number of neurons in each layer and name of variable they predict. name modelA_32_16_16_8 could be decoded as model that has input layer, 4 hidden layers with 32, 16, 16 and 8 neurons respectively and predicts variable A. 

deprecated:
    folder contains deprecated code. parts of the code will be rewritten and used to write thesis. folder will be probably deleted after thesis is written. 

datasets:
    folder contains different datasets generated with VWASE.

training_sample:
    notebook contains class that describes and contains functions that perform actions on single sample meant for training. sample contains measurements of PSI and DELTA for wavelengths ranging from 300nm to 1000nm and for 3 different angles of incidence. 

training_data:
    notebook contains class that describes and contains functions that perform actions on training data. it is constituded by list of samples and contains functions that perform actions on whole set of data. from this class training can be performed and scalers for different variables can be generated. 

model_training:
    it is used to train the model. it saves model version with lowest loss value. it allows you to choose optimizer and leraning rate.

model_creator:
    it is used to create the model that is used for training. it allows you to determine the number of neurons in each layer and the activation function. loss is set to MSE by default.

working_notebook:
    contains code written using newly created classes. it is used to test the code and check if everything works as expected.
