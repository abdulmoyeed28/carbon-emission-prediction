import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels
from scipy import stats
import statsmodels.api as sm
from scipy.stats import shapiro
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import Lasso, Ridge, ElasticNet, SGDRegressor, LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, cross_val_score, train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format

def compareAlgorithms():
    data = pd.read_csv("../CO2 Emissions_Canada.csv")
    data = data.rename(columns={
        'Vehicle Class': 'Vehicle_Class',
        'Engine Size(L)': 'Engine_Size',
        'Fuel Type': 'Fuel_Type',
        'Fuel Consumption City (L/100 km)': 'Fuel_Consumption_City',
        'Fuel Consumption Hwy (L/100 km)': 'Fuel_Consumption_Hwy',
        'Fuel Consumption Comb (L/100 km)': 'Fuel_Consumption_Comb',
        'Fuel Consumption Comb (mpg)': 'Fuel_Consumption_Comb1',
        'CO2 Emissions(g/km)': 'CO2_Emissions'
    })

    data.drop_duplicates(inplace=True)

    data.reset_index(inplace=True, drop=True)

    data_num_features = data.select_dtypes(include=np.number)
    print('The numerical columns in the dataset are: ', data_num_features.columns)

    data['Make_Type'] = data['Make'].replace(['BUGATTI', 'PORSCHE', 'MASERATI', 'ASTON MARTIN', 'LAMBORGHINI', 'JAGUAR','SRT'], 'Sports')
    data['Make_Type'] = data['Make_Type'].replace(['ALFA ROMEO', 'AUDI', 'BMW', 'BUICK', 'CADILLAC', 'CHRYSLER', 'DODGE', 'GMC','INFINITI', 'JEEP', 'LAND ROVER', 'LEXUS', 'MERCEDES-BENZ','MINI', 'SMART', 'VOLVO'],'Premium')
    data['Make_Type'] = data['Make_Type'].replace(['ACURA', 'BENTLEY', 'LINCOLN', 'ROLLS-ROYCE',  'GENESIS'], 'Luxury')
    data['Make_Type'] = data['Make_Type'].replace(['CHEVROLET', 'FIAT', 'FORD', 'KIA', 'HONDA', 'HYUNDAI', 'MAZDA', 'MITSUBISHI','NISSAN', 'RAM', 'SCION', 'SUBARU', 'TOYOTA','VOLKSWAGEN'],'General')

    data.drop(['Make'], inplace=True, axis=1)

    data['Vehicle_Class_Type'] = data['Vehicle_Class'].replace(['COMPACT', 'MINICOMPACT', 'SUBCOMPACT'], 'Hatchback')
    data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(
        ['MID-SIZE', 'TWO-SEATER', 'FULL-SIZE', 'STATION WAGON - SMALL', 'STATION WAGON - MID-SIZE'], 'Sedan')
    data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(['SUV - SMALL', 'SUV - STANDARD', 'MINIVAN'], 'SUV')
    data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(
        ['VAN - CARGO', 'VAN - PASSENGER', 'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE',
         'PICKUP TRUCK - SMALL'], 'Truck')

    data.drop(['Vehicle_Class'], inplace=True, axis=1)

    data.drop(['Model'], axis=1, inplace=True)

    df_num_features = data.select_dtypes(include=np.number)
    Q1 = df_num_features.quantile(0.25)
    Q3 = df_num_features.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    outlier = pd.DataFrame((df_num_features < (Q1 - 1.5 * IQR)) | (df_num_features > (Q3 + 1.5 * IQR)))

    for i in outlier.columns:
        print('Total number of Outliers in column {} are {}'.format(i, (len(outlier[outlier[i] == True][i]))))

    stat, p_value = shapiro(df_num_features)

    print('Test statistic:', stat)
    print('P-Value:', p_value)
    ###
    # Align Q1, Q3, and IQR with data's index
    Q1, Q3, IQR = Q1.align(data, axis=0), Q3.align(data, axis=0), IQR.align(data, axis=0)

    # Proceed with the filtering operation
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    ###
    #data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    data.reset_index(inplace=True, drop=True)

    df_dummies = pd.get_dummies(data=data[["Fuel_Type", "Transmission", "Make_Type", "Vehicle_Class_Type"]],
                                drop_first=True)

    df_num_features = data.select_dtypes(include=np.number)

    df_comb = pd.concat([df_num_features, df_dummies], axis=1)

    X = df_comb.drop(['CO2_Emissions'], axis=1)
    y = df_comb['CO2_Emissions']

    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    df_num_features.skew()

    for col in df_num_features.columns:
        print("Column ", col, " :", stats.shapiro(df_num_features[col]))

    df_num_features.drop('CO2_Emissions', axis=1, inplace=True)

    mms = MinMaxScaler()
    mmsfit = mms.fit(df_num_features)
    dfx = pd.DataFrame(mms.fit_transform(df_num_features), columns=['Engine_Size', 'Cylinders', 'Fuel_Consumption_City',
                                                                    'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb',
                                                                    'Fuel_Consumption_Comb1'])

    df = pd.concat([dfx, df_dummies], axis=1)

    X = df.copy()
    X = sm.add_constant(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    MLR_model2 = sm.OLS(y_train, X_train).fit()
    MLR_model2.summary()

    target = df_comb['CO2_Emissions']

    vif = pd.DataFrame()
    vif["VIF_Factor"] = [variance_inflation_factor(df_num_features.values, i) for i in range(df_num_features.shape[1])]
    vif["Features"] = df_num_features.columns
    vif.sort_values('VIF_Factor', ascending=False).reset_index(drop=True)

    sklearn_pca = PCA()
    pcafit = sklearn_pca.fit(df)

    pcafit.explained_variance_

    plt.plot(np.cumsum(pcafit.explained_variance_ratio_))
    plt.locator_params(axis="x", nbins=len(pcafit.explained_variance_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.savefig('../static/vis/cev.jpg')
    plt.clf()

    df_pca = sklearn_pca.fit_transform(df)
    df_pca = pd.DataFrame(df_pca, columns=['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5',
                                           'pca6', 'pca7', 'pca8', 'pca9', 'pca10', 'pca11',
                                           'pca12', 'pca13', 'pca14', 'pca15', 'pca16',
                                           'pca17', 'pca18', 'pca19', 'pca20', 'pca21', 'pca22',
                                           'pca23', 'pca24', 'pca25', 'pca26', 'pca27', 'pca28',
                                           'pca29', 'pca30', 'pca31', 'pca32', 'pca33',
                                           'pca34', 'pca35', 'pca36', 'pca37', 'pca38', 'pca39',
                                           'pca40', 'pca41'])

    df_pca = sm.add_constant(df_pca)

    X = df_pca[
        ['const', 'pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10', 'pca11',
         'pca12', 'pca13', 'pca14', 'pca15', 'pca16', 'pca17', 'pca18', 'pca19', 'pca20', 'pca21', 'pca22', 'pca23',
         'pca24', 'pca25', 'pca26', 'pca27', 'pca28', 'pca29', 'pca30', 'pca31', 'pca32', 'pca33']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    MLR_model_pca = sm.OLS(y_train, X_train).fit()
    MLR_model_pca.summary()

    linreg = LinearRegression()
    linreg_forward = sfs(estimator=linreg, k_features='best', forward=True,
                         verbose=0, scoring='r2')

    sfs_forward = linreg_forward.fit(X_train, y_train)

    print('Features selected using forward selection are: ')
    print(sfs_forward.k_feature_names_)

    print('\nR-Squared: ', sfs_forward.k_score_)

    linreg = LinearRegression()
    linreg_backward = sfs(estimator=linreg, k_features='best', forward=False,
                          verbose=0, scoring='r2')

    sfs_backward = linreg_backward.fit(X_train, y_train)

    print('Features selected using backward elimination are: ')
    print(sfs_backward.k_feature_names_)

    print('\nR-Squared: ', sfs_backward.k_score_)

    X = df_pca[
        ['const', 'pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10', 'pca11',
         'pca12', 'pca13', 'pca14', 'pca15', 'pca16', 'pca17', 'pca18', 'pca19', 'pca20', 'pca21', 'pca23', 'pca24',
         'pca25', 'pca26', 'pca27', 'pca28', 'pca29', 'pca30', 'pca31', 'pca32', 'pca33']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    MLR_full_model = sm.OLS(y_train, X_train).fit()
    MLR_full_model.summary()

    name = ['f-value', 'p-value']
    test = sms.het_breuschpagan(MLR_full_model.resid, MLR_full_model.model.exog)
    lzip(name, test[2:])

    plt.rcParams['figure.figsize'] = [15, 8]

    qqplot(MLR_full_model.resid, line='r')

    plt.title('Q-Q Plot', fontsize=15)
    plt.xlabel('Theoretical Quantiles', fontsize=15)
    plt.ylabel('Sample Quantiles', fontsize=15)
    plt.savefig('../static/vis/qqplot.jpg')
    plt.clf()

    stat, p_value = shapiro(MLR_full_model.resid)
    print('Test statistic:', stat)
    print('P-Value:', p_value)

    y_train_pred = MLR_full_model.predict(X_train)
    y_train_pred.head()

    ssr = np.sum((y_train_pred - y_train.mean()) ** 2)
    sse = np.sum((y_train - y_train_pred) ** 2)
    sst = np.sum((y_train - y_train.mean()) ** 2)

    print('Sum of Squared Regression:', ssr)
    print('Sum of Squared Error:', sse)
    print('Sum of Sqaured Total:', sst)
    print('Sum of SSR and SSE is:', ssr + sse)

    r_sq = MLR_full_model.rsquared

    # print the R-squared value
    print('R Squared is:', r_sq)

    see = np.sqrt(sse / (len(X_train) - 2))
    print("The standard error of estimate:", see)

    t_intercept = MLR_full_model.params[0] / MLR_full_model.bse[0]
    print('t intercept:', t_intercept)

    t_coeff1 = MLR_full_model.params[1] / MLR_full_model.bse[1]
    print('t coeff:', t_coeff1)

    pval = stats.t.sf(np.abs(t_intercept), 4069) * 2
    print('p val for intercept:', pval)

    CI_inter_min, CI_inter_max = MLR_full_model.params[0] - (1.9622 * MLR_full_model.bse[0]), MLR_full_model.params[
        0] + (1.9622 * MLR_full_model.bse[0])
    print('CI for intercept:', [CI_inter_min, CI_inter_max])

    CI_coeff1_min, CI_coeff1_max = MLR_full_model.params[1] - (1.9622 * MLR_full_model.bse[1]), MLR_full_model.params[
        1] + (1.9622 * MLR_full_model.bse[1])
    print('CI for coeff1:', [CI_coeff1_min, CI_coeff1_max])

    r_sq_mlr = MLR_full_model.rsquared
    print('r square in regression model:', r_sq_mlr)

    adj_r_sq = MLR_full_model.rsquared_adj
    print('Adjusted r square for regression model:', adj_r_sq)

    k = len(X_train.columns)
    n = len(X_train)

    f_value = (r_sq_mlr / (k - 1)) / ((1 - r_sq_mlr) / (n - k))
    print('f value for regression model:', f_value)

    p_val = stats.f.sf(f_value, dfn=31, dfd=4364)
    print('p value for regression model:', p_val)

    train_pred = MLR_full_model.predict(X_train)
    test_pred = MLR_full_model.predict(X_test)

    mse_train = round(mean_squared_error(y_train, train_pred), 4)
    mse_test = round(mean_squared_error(y_test, test_pred), 4)

    print("Mean Squared Error (MSE) on training set: ", mse_train)
    print("Mean Squared Error (MSE) on test set: ", mse_test)

    rmse_train = round(np.sqrt(mse_train), 4)

    mse_test = mean_squared_error(y_test, test_pred)
    rmse_test = round(np.sqrt(mse_test), 4)

    print("Root Mean Squared Error (RMSE) on training set: ", rmse_train)
    print("Root Mean Squared Error (RMSE) on test set: ", rmse_test)

    mae_train = round(mean_absolute_error(y_train, train_pred), 4)
    mae_test = round(mean_absolute_error(y_test, test_pred), 4)

    print("Mean Absolute Error (MAE) on training set: ", mae_train)
    print("Mean Absolute Error (MAE) on test set: ", mae_test)

    def mape(actual, predicted):
        return (np.mean(np.abs((actual - predicted) / actual)) * 100)

    mape_train = round(mape(y_train, train_pred), 4)
    mape_test = round(mape(y_test, test_pred), 4)

    print("Mean Absolute Percentage Error (MAPE) on training set: ", mape_train)
    print("Mean Absolute Percentage Error (MAPE) on test set: ", mape_test)

    cols = ['Model_Name', 'R-squared', 'Adj. R-squared', 'MSE', 'RMSE', 'MAE', 'MAPE']

    result_table = pd.DataFrame(columns=cols)

    MLR_full_model_metrics = pd.Series({'Model_Name': "MLR Full Model",
                                        'R-squared': MLR_full_model.rsquared,
                                        'Adj. R-squared': MLR_full_model.rsquared_adj,
                                        'MSE': mean_squared_error(y_test, test_pred),
                                        'RMSE': rmse(y_test, test_pred),
                                        'MAE': mean_absolute_error(y_test, test_pred),
                                        'MAPE': mape(y_test, test_pred)
                                        })

    result_table = result_table.append(MLR_full_model_metrics, ignore_index=True)

    kf = KFold(n_splits=10)

    def Get_score(model, X_train_k, X_test_k, y_train_k, y_test_k):
        model.fit(X_train_k, y_train_k)
        return model.score(X_test_k, y_test_k)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)

    scores = []

    for train_index, test_index in kf.split(X_train):
        X_train_k, X_test_k, y_train_k, y_test_k = X_train.iloc[train_index], X_train.iloc[test_index], \
                                                   y_train.iloc[train_index], y_train.iloc[test_index]

        scores.append(Get_score(LinearRegression(), X_train_k, X_test_k, y_train_k, y_test_k))

    print('All scores: ', scores)
    print("\nMinimum score obtained: ", round(min(scores), 4))
    print("Maximum score obtained: ", round(max(scores), 4))
    print("Average score obtained: ", round(np.mean(scores), 4))

    scores = cross_val_score(estimator=LinearRegression(),
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='r2')

    print('All scores: ', scores)
    print("\nMinimum score obtained: ", round(min(scores), 4))
    print("Maximum score obtained: ", round(max(scores), 4))
    print("Average score obtained: ", round(np.mean(scores), 4))

    def Get_score(model, X_train_k, X_test_k, y_train_k, y_test_k):
        model.fit(X_train_k, y_train_k)
        return model.score(X_test_k, y_test_k)

    loocv_rmse = []
    loocv = LeaveOneOut()

    for train_index, test_index in loocv.split(X_train):
        X_train_l, X_test_l, y_train_l, y_test_l = X_train.iloc[train_index], X_train.iloc[test_index], \
                                                   y_train.iloc[train_index], y_train.iloc[test_index]

        linreg = LinearRegression()
        linreg.fit(X_train_l, y_train_l)

        mse = mean_squared_error(y_test_l, linreg.predict(X_test_l))
        rmse = np.sqrt(mse)
        loocv_rmse.append(rmse)

    print("\nMinimum rmse obtained: ", round(min(loocv_rmse), 4))
    print("Maximum rmse obtained: ", round(max(loocv_rmse), 4))
    print("Average rmse obtained: ", round(np.mean(loocv_rmse), 4))

    def get_train_rmse(model):

        train_pred = model.predict(X_train)
        mse_train = mean_squared_error(y_train, train_pred)
        rmse_train = round(np.sqrt(mse_train), 4)
        return (rmse_train)

    def get_test_rmse(model):

        test_pred = model.predict(X_test)
        mse_test = mean_squared_error(y_test, test_pred)
        rmse_test = round(np.sqrt(mse_test), 4)
        return (rmse_test)

    def get_test_mape(model):

        test_pred = model.predict(X_test)
        mape_test = mape(y_test, test_pred)
        return (mape_test)

    def plot_coefficients(model, algorithm_name):

        df_coeff = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
        sorted_coeff = df_coeff.sort_values('Coefficient', ascending=False)
        sns.barplot(x="Coefficient", y="Variable", data=sorted_coeff)
        plt.xlabel("Coefficients from {}".format(algorithm_name), fontsize=15)
        plt.ylabel('Features', fontsize=15)

    def get_score(model):

        r_sq = model.score(X_train, y_train)
        n = X_train.shape[0]
        k = X_train.shape[1]
        r_sq_adj = 1 - ((1 - r_sq) * (n - 1) / (n - k - 1))
        return ([r_sq, r_sq_adj])

    sgd = SGDRegressor(random_state=10)
    linreg_with_SGD = sgd.fit(X_train, y_train)

    print('RMSE on train set:', get_train_rmse(linreg_with_SGD))
    print('RMSE on test set:', get_test_rmse(linreg_with_SGD))

    MLR_model = linreg.fit(X_train, y_train)

    score_card = pd.DataFrame(columns=['Model_Name', 'Alpha (Wherever Required)', 'l1-ratio', 'R-Squared',
                                       'Adj. R-Squared', 'Train_RMSE', 'Test_RMSE', 'Test_MAPE'])

    def update_score_card(algorithm_name, model, alpha='-', l1_ratio='-'):

        global score_card
        score_card = score_card.append({'Model_Name': algorithm_name,
                                        'Alpha (Wherever Required)': alpha,
                                        'l1-ratio': l1_ratio,
                                        'Test_MAPE': get_test_mape(model),
                                        'Train_RMSE': get_train_rmse(model),
                                        'Test_RMSE': get_test_rmse(model),
                                        'R-Squared': get_score(model)[0],
                                        'Adj. R-Squared': get_score(model)[1]}, ignore_index=True)

    update_score_card(algorithm_name='Linear Regression (using SGD)', model=linreg_with_SGD)

    print(score_card)

    ridge = Ridge(alpha=0.1, max_iter=500)
    ridge.fit(X_train, y_train)

    update_score_card(algorithm_name='Ridge Regression (with alpha = 0.1)', model=ridge, alpha=0.1)

    print('RMSE on test set:', get_test_rmse(ridge))

    ridge = Ridge(alpha=1, max_iter=500)
    ridge.fit(X_train, y_train)

    update_score_card(algorithm_name='Ridge Regression (with alpha = 1)', model=ridge, alpha=1)

    print('RMSE on test set:', np.round(get_test_rmse(ridge), 2))

    ridge = Ridge(alpha=2, max_iter=500)
    ridge.fit(X_train, y_train)

    update_score_card(algorithm_name='Ridge Regression (with alpha = 2)', model=ridge, alpha=2)

    print('RMSE on test set:', get_test_rmse(ridge))

    ridge = Ridge(alpha=0.5, max_iter=500)
    ridge.fit(X_train, y_train)

    update_score_card(algorithm_name='Ridge Regression (with alpha = 0.5)', model=ridge, alpha=0.5)

    print('RMSE on test set:', get_test_rmse(ridge))

    lasso = Lasso(alpha=0.01, max_iter=500)
    lasso.fit(X_train, y_train)

    print('RMSE on test set:', get_test_rmse(lasso))

    lasso = Lasso(alpha=0.05, max_iter=500)
    lasso.fit(X_train, y_train)

    print('RMSE on test set:', get_test_rmse(lasso))

    df_lasso_coeff = pd.DataFrame({'Variable': X.columns, 'Coefficient': lasso.coef_})

    print('Insignificant variables obtained from Lasso Regression when alpha is 0.05')
    df_lasso_coeff.Variable[df_lasso_coeff.Coefficient == 0].to_list()
    update_score_card(algorithm_name='Lasso Regression', model=lasso, alpha='0.05')

    enet = ElasticNet(alpha=0.1, l1_ratio=0.55, max_iter=500)
    enet.fit(X_train, y_train)

    update_score_card(algorithm_name='Elastic Net Regression', model=enet, alpha='0.1', l1_ratio='0.55')

    print('RMSE on test set:', get_test_rmse(enet))

    enet = ElasticNet(alpha=0.1, l1_ratio=0.1, max_iter=500)
    enet.fit(X_train, y_train)

    update_score_card(algorithm_name='Elastic Net Regression', model=enet, alpha='0.1', l1_ratio='0.1')

    print('RMSE on test set:', get_test_rmse(enet))
    enet = ElasticNet(alpha=0.1, l1_ratio=0.01, max_iter=500)
    enet.fit(X_train, y_train)

    update_score_card(algorithm_name='Elastic Net Regression', model=enet, alpha='0.1', l1_ratio='0.01')

    print('RMSE on test set:', get_test_rmse(enet))

    tuned_paramaters = [{'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1, 5, 10, 20, 40, 60, 80, 100]}]

    ridge = Ridge()
    ridge_grid = GridSearchCV(estimator=ridge,
                              param_grid=tuned_paramaters,
                              cv=10)

    ridge_grid.fit(X_train, y_train)

    print('Best parameters for Ridge Regression: ', ridge_grid.best_params_, '\n')
    print('RMSE on test set:', get_test_rmse(ridge_grid))

    update_score_card(algorithm_name='Ridge Regression (using GridSearchCV)',
                      model=ridge_grid,
                      alpha=ridge_grid.best_params_.get('alpha'))

    tuned_paramaters = [{'alpha': [1e-15, 1e-10, 1e-8, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20]}]

    lasso = Lasso()
    lasso_grid = GridSearchCV(estimator=lasso,
                              param_grid=tuned_paramaters,
                              cv=10)

    lasso_grid.fit(X_train, y_train)

    print('Best parameters for Lasso Regression: ', lasso_grid.best_params_, '\n')
    print('RMSE on test set:', get_test_rmse(lasso_grid))

    update_score_card(algorithm_name='Lasso Regression (using GridSearchCV)',
                      model=lasso_grid,
                      alpha=lasso_grid.best_params_.get('alpha'))

    tuned_paramaters = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 20, 40, 60],
                         'l1_ratio': [0.0001, 0.0002, 0.001, 0.01, 0.1, 0.2, 0.4, 0.55]}]

    enet = ElasticNet()
    enet_grid = GridSearchCV(estimator=enet,
                             param_grid=tuned_paramaters,
                             cv=10)

    enet_grid.fit(X_train, y_train)

    print('Best parameters for Elastic Net Regression: ', enet_grid.best_params_, '\n')
    print('RMSE on test set:', get_test_rmse(enet_grid))

    update_score_card(algorithm_name='Elastic Net Regression (using GridSearchCV)',
                      model=enet_grid,
                      alpha=enet_grid.best_params_.get('alpha'),
                      l1_ratio=enet_grid.best_params_.get('l1_ratio'))

    score_card = score_card.sort_values('Test_RMSE').reset_index(drop=True)
    score_card.style.highlight_min(color='lightblue', subset='Test_RMSE')

#compareAlgorithms()


