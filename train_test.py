import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras import models
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def merck_dnn_model(input_shape=(128)):
    """
    # The recommended network presented in the paper: Junshui Ma et. al., Deep Neural Nets as a Method for Quantitative 
    # Structure Activity Relationships
    # URL: http://www.cs.toronto.edu/~gdahl/papers/deepQSARJChemInfModel2015.pdf
    # :param input_shape: dim of input features
    # :return: a keras model
    """
    
    model = Sequential()
    
    model.add(Dense(4000, activation='relu', input_dim = 4730, kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(2000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu',  kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.25))

    model.add(Dense(1000, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dropout(0.10))

    model.add(Dense(1, activation=None, use_bias=True, kernel_regularizer=l2(0.0001)))
    
    model.compile(loss='mean_squared_error', optimizer='sgd')
    
    return model

def random_forest_model():
    return RandomForestRegressor(n_estimators = 100, random_state = 0) 

def xgb_model(max_depth = 8, n_estimators = 150):
    xgb_model = XGBRegressor(max_depth, n_estimators,  silent = False, verbose = True)
    return xgb_model

def get_file_path(i):
    return current_path + '/data/TrainingSet/ACT' + str(i) +'_competition_training.csv'

def train_test_split(training_data, training_proportion):
    np.random.seed(123)
    msk = np.random.rand(training_data.shape[0]) < training_proportion
    relevant_columns = list(training_data.columns)
    relevant_columns.remove('MOLECULE')
    train = training_data[msk][relevant_columns]
    test = training_data[~msk][relevant_columns]
    return train, test

def train_model(train, model):
    relevant_columns = list(train.columns)
    relevant_columns.remove('Act')
    X = train[relevant_columns]
    y = train['Act']
    model.fit(X, y, verbose=True)
    return xgb_model

def predict_test(test, model):
    relevant_columns = list(train.columns)
    relevant_columns.remove('Act')
    X_test = test[relevant_columns]
    test['predictions'] = model.predict(X_test)
    return test

def evaluation(test, col1, col2):
    return {'count': test.shape[0],
            'r2': r2_score(test[col1], test[col2]),
            'rmse': np.sqrt(mean_squared_error(test[col1], test[col2])),
            'mae': np.abs(test[col2]-test[col1]).mean(),
            'col1_mean': test[col1].mean(),
            'col2_mean': test[col2].mean(),
            'number_of_columns':test.shape[1]}