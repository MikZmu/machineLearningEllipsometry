import torch
import statistics


def predict_file_median(model, file_dataset):

    values = []

    for i in file_dataset:

        with torch.no_grad():
            prediction = model(i)
            prediction = prediction.item()
            values.append(prediction)

    return statistics.median(values)


def predict_file_mean(model, file_dataset):

    values = []

    for i in file_dataset:
        #print(i)
        with torch.no_grad():
            prediction = model(i)
            prediction = prediction.item()
            values.append(prediction)

    return statistics.mean(values)

