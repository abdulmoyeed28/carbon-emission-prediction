import pickle
#import jsonify
#import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def createModel():
    data = pd.read_csv('final_co2.csv', index_col=0)

    X = data.drop(['CO2_Emissions'], axis=1)
    y = data['CO2_Emissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = round(r2_score(y_test, y_pred) * 100, 3)


    filename = '../finalized_model.pkl'
    pickle.dump(reg, open(filename, 'wb'))

    msg="Linear Regression Model created successfully"

    return msg, r2


#createModel()
