import os
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

def prepare_data():
  train = pd.read_csv(r"C:\Users\Elena\PycharmProjects\HW9\data\realty_data.csv")
  train = train.drop(columns=['product_name','period', 'postcode', 'address_name', 'lat', 'lon', 'object_type',   'city', 'settlement',
'district', 'area', 'description', 'source'])
  train.dropna(inplace=True)
  return train

def train_model(train):
  X = train[['total_square',	'rooms',	'floor']]
  y = train['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = LinearRegression()
  model.fit(X_train, y_train)

  with open('rf_fitted.pkl', 'wb') as file:
      pickle.dump(model, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model