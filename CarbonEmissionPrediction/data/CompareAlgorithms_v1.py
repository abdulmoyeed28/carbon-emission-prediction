import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


r2={}


def compAlg():
    data = pd.read_csv('final_co2.csv', index_col=0)

    X = data.drop(['CO2_Emissions'], axis=1)
    y = data['CO2_Emissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['LR'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) LR: {0:.3f}'.format(r2['LR']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/LR_pred.jpg')
    plt.clf()

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['RF'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) RF: {0:.3f}'.format(r2['RF']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/RF_pred.jpg')
    plt.clf()

    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['DT'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) DT: {0:.3f}'.format(r2['DT']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/DT_pred.jpg')
    plt.clf()

    reg = XGBRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['XGB'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) XGB: {0:.3f}'.format(r2['XGB']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/XGB_pred.jpg')
    plt.clf()

    reg = AdaBoostRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['AB'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) AB: {0:.3f}'.format(r2['AB']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/AB_pred.jpg')
    plt.clf()

    reg = Lasso(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2['Lasso'] = round(r2_score(y_test, y_pred)*100,3)
    print('R2_score (test) Lasso: {0:.3f}'.format(r2['Lasso']))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')  # Diagonal line
    plt.xlabel("Actual CO2 Emissions")
    plt.ylabel("Predicted CO2 Emissions")
    plt.title("Actual vs Predicted Values")
    plt.savefig('static/vis/Lasso_pred.jpg')
    plt.clf()

    # Plotting accuracies of all the models
    colors = ["green", "yellow", "black", "magenta", "#0e76a8", "red"]

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel("\nAccuracy %", fontsize=20)
    plt.xlabel("\nAlgorithms", fontsize=20)
    sns.barplot(x=list(r2.keys()), y=list(r2.values()), palette=colors)

    ax = sns.barplot(x=list(r2.keys()), y=list(r2.values()), palette=colors)
    for i, accuracy in enumerate(r2.values()):
        plt.text(i, accuracy + 2, str(accuracy), ha='center', va='bottom', fontsize=10)

    plt.savefig('static/vis/AlgComp.jpg')
    plt.clf()

    return r2


#compAlg()






