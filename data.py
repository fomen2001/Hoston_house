# data.py
import numpy as np
from keras.datasets import boston_housing

def load_and_preprocess_data():
    # Chargement des données
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    # Normalisation des données
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    
    return (x_train_norm, y_train), (x_test_norm, y_test)
