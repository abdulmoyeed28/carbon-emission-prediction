import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format

def dataAnalysis():
    data = pd.read_csv("CO2 Emissions_Canada.csv")
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

    plt.figure(figsize=(20, 10))

    corr = data_num_features.corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap='tab20c')
    plt.savefig("static/vis/corr.jpg")
    plt.clf()

    data_cat_features = data.select_dtypes(include='object')
    data_cat_features1 = data_cat_features[['Vehicle_Class', 'Transmission', 'Fuel_Type', 'Model', 'Make']]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    cat_count = data['Vehicle_Class'].value_counts()
    cat_count10 = cat_count[:10, ]
    #z = sns.barplot(cat_count10.values, cat_count10.index, alpha=0.8, ax=ax)
    z = sns.barplot(x=cat_count10.values, y=cat_count10.index, alpha=0.8, ax=ax, palette='tab10')
    if cat_count.size > 10:
        z.set_title('Top 10 Vehicle_Class')
    else:
        z.set_title('Vehicle_Class')
    z.set_xlabel('Number of Cars', fontsize=9)
    plt.tight_layout()
    plt.savefig("static/vis/veh_cls.jpg")
    plt.clf()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    cat_count = data['Transmission'].value_counts()
    cat_count10 = cat_count[:10, ]
    # z = sns.barplot(cat_count10.values, cat_count10.index, alpha=0.8, ax=ax)
    z = sns.barplot(x=cat_count10.values, y=cat_count10.index, alpha=0.8, ax=ax, palette='tab10')
    if cat_count.size > 10:
        z.set_title('Top 10 Vehicle_Class')
    else:
        z.set_title('Transmission')
    z.set_xlabel('Number of Cars', fontsize=9)
    plt.tight_layout()
    plt.savefig("static/vis/trans.jpg")
    plt.clf()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    cat_count = data['Fuel_Type'].value_counts()
    cat_count10 = cat_count[:10, ]
    # z = sns.barplot(cat_count10.values, cat_count10.index, alpha=0.8, ax=ax)
    z = sns.barplot(x=cat_count10.values, y=cat_count10.index, alpha=0.8, ax=ax, palette='tab10')
    if cat_count.size > 10:
        z.set_title('Top 10 Fuel_Type')
    else:
        z.set_title('Fuel_Type')
    z.set_xlabel('Number of Cars', fontsize=9)
    plt.tight_layout()
    plt.savefig("static/vis/ftype.jpg")
    plt.clf()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    cat_count = data['Model'].value_counts()
    cat_count10 = cat_count[:10, ]
    # z = sns.barplot(cat_count10.values, cat_count10.index, alpha=0.8, ax=ax)
    z = sns.barplot(x=cat_count10.values, y=cat_count10.index, alpha=0.8, ax=ax, palette='tab10')
    if cat_count.size > 10:
        z.set_title('Top 10 Model')
    else:
        z.set_title('Model')
    z.set_xlabel('Number of Cars', fontsize=9)
    plt.tight_layout()
    plt.savefig("static/vis/model.jpg")
    plt.clf()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    cat_count = data['Make'].value_counts()
    cat_count10 = cat_count[:10, ]
    # z = sns.barplot(cat_count10.values, cat_count10.index, alpha=0.8, ax=ax)
    z = sns.barplot(x=cat_count10.values, y=cat_count10.index, alpha=0.8, ax=ax, palette='tab10')
    if cat_count.size > 10:
        z.set_title('Top 10 Make')
    else:
        z.set_title('Make')
    z.set_xlabel('Number of Cars', fontsize=9)
    plt.tight_layout()
    plt.savefig("static/vis/make.jpg")
    plt.clf()



#dataAnlysis()




