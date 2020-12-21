import numpy as np
import pandas as pd
from random import choice
from tensorflow.keras.models import model_from_json


def load_model_from_json(model_name):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    return loaded_model

def save_model_to_json(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + '.h5')


def normalize_data(data, minimum = None,maximum = None):
    if minimum is None:
        minimum = np.min(np.min(data))
    if maximum is None:
        maximum = np.max(np.max(data))
    data_norm = (data - minimum) / (maximum - minimum)
    return minimum, maximum, data_norm


def denormalize_data(data, minimum, maximum):
    return data * (maximum - minimum) + minimum

def save_csv(data,header,path='test.csv'):
    if isinstance(header,list):
        header = ','.join(header)
    np.savetxt(path,data,delimiter=',',fmt='%s',header= header,comments='')

def load_csv(path):
    df = pd.read_csv(path)
    header = df.columns.to_list()
    data = df.to_numpy()
    return header, data

def get_random_features(features, size):
    headers = list(features.keys())
    samples = []
    for _ in range(size):
        samples.append( [choice(arr) for arr in features.values()] )
    
    return headers, samples
    
def is_in_range(value,minimum,maximum):
    return (value >= minimum) and (value <= maximum)
