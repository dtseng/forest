"""
Example of how to train/predict using the forest module.
"""


import numpy as np
import random
import scipy.io as sio
import pandas as pd
import forest
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Replace NaN with median if noncategorical, otherwise mode"""
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def shuffle(X, Y):
    indices = np.arange(X.shape[0])
    random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

def loadSpam():
    data = sio.loadmat('spam_data/spam_data.mat')
    Xtest = np.array(data['test_data'])
    Xtrain = np.array(data['training_data'])
    ytrain = np.array(data['training_labels'])
    return Xtest, Xtrain, ytrain

def loadCensus():
    dataframe = pd.read_csv("census_data/train_data.csv", header=0)
    y = dataframe['label'].values
    everything = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    categories = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    non_categories = [x for x in everything if x not in categories]
    modified_data = dataframe[everything].replace(to_replace='?',value=np.nan) # Replace ? with NaN, then replace NaN with preprocessing
    modified_data = DataFrameImputer().fit_transform(modified_data) # one-hot encoding
    numerics = modified_data[non_categories].values
    v = DictVectorizer(sparse=False)
    temp = modified_data[categories].transpose().to_dict().values()
    categorical_numpy = v.fit_transform(temp)
    xtrain = np.concatenate((categorical_numpy, numerics),axis=1)
    dftest = pd.read_csv("census_data/test_data.csv", header=0)
    modified_test_data = dftest[everything].replace(to_replace='?', value=np.nan)
    modified_test_data = DataFrameImputer().fit_transform(modified_test_data)
    numerical_data_test = modified_test_data[non_categories].values
    temp = modified_test_data[categories].transpose().to_dict().values()
    categorical_data_test = v.transform(temp)
    xtest = np.concatenate((categorical_data_test, numerical_data_test), axis=1)
    return xtrain, y, xtest, non_categories, v.get_feature_names()

Xtrain, ytrain, Xtest, numericalnames, feature_names = loadCensus()

ytrain = ytrain.reshape(ytrain.size)
Xtrain, ytrain = shuffle(Xtrain, ytrain)
cutoff = int(0.7*Xtrain.shape[0])

# Split the loaded data into training/validation sets.
actualXtrain = Xtrain[0:cutoff]
actualytrain = ytrain[0:cutoff]

validationX = Xtrain[cutoff:]
validationY = ytrain[cutoff:]


randomforest = forest.RandomForest(40, 0.2, actualXtrain, actualytrain)
print("forest train:")
prediction = randomforest.predict(actualXtrain)
randomforest.print_accuracy(prediction, actualytrain)
print("forest validation:")
prediction = randomforest.predict(validationX)
randomforest.print_accuracy(prediction, validationY)
