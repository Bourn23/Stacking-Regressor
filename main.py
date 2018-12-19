from random import randint
import pandas as pd
import numpy as np

#Loading Models
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# Loading Train & Test dataset
df_train = pd.read_csv('001_train.csv')
df_test = pd.read_csv('001_test.csv')


# Predicators & Target columns
predicators = ['Item_Weight', 'Item_Fat_Content',
       'Item_MRP', 'Item_Visibility', 'Items_ID',
        'Outlet_Size','Outlet_Age', 'Outlet_Location_Type', 'Outlet_Type' ]
target = 'Item_Outlet_Sales'

# Choosing base models
base_models = [DecisionTreeRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, XGBRegressor]

# Setting up train dataset for ensemble model
row_train = len(df_train)
col_train = len(base_models)
stacking_train_dataset1 = np.zeros(shape = (row_train, col_train))
stacking_train_dataset = pd.DataFrame(data= stacking_train_dataset1)


# Setting up test dataset for ensemble model
row_test = len(df_test)
col_test = len(base_models)
stacking_test_dataset1 = np.zeros(shape = (row_test, col_test))
stacking_test_dataset = pd.DataFrame(data = stacking_test_dataset1)


# Setting up KFold splitter
number_of_split = 10
split = KFold(n_splits = number_of_split, random_state = 42, shuffle = True)

#Training Model and Filling up Train & Test dataset for stacking ensemble
for i, base_alg in enumerate(base_models):
    #using counter & inner_counter randomly chooses one of the split from dataset
    #allowing each model being trained on different dataset
    #different algorithm with different datasets
    counter = randint(0, number_of_split)
    inner_counter = 0

    for trainix , testix in split.split(stacking_train_dataset):
        if inner_counter == counter:
            x_train = df_train[predicators].iloc[trainix]
            y_train = df_train[target].iloc[trainix] 

            model = base_alg()
            print('training : %s'%(type(model)))
            model.fit(x_train, y_train)

            #Filling up train & test datasets (CV) with the model predictions
            stacking_train_dataset[i] = model.predict(df_train[predicators])
            stacking_test_dataset[i] = model.predict(df_test[predicators])

        inner_counter += 1


# You can change the ensemble model here
ensemble_model = LinearRegression()


# Ensemble Model Prediction
stacking_train_dataset['Ensemble'] = ensemble_model.fit(stacking_train_dataset, df_train[target]).predict(stacking_train_dataset)

# Preparing for sumbission
IDcol = ['Item_Identifier', 'Outlet_Identifier']
submission = df_test[IDcol]
submission[target] = ensemble_model.predict(stacking_test_dataset)
submission.to_csv('Submit01.csv', index = False)

stacking_test_dataset['Ensemble_Output'] = ensemble_model.predict(stacking_test_dataset)

# Saving Created Datasets
stacking_test_dataset.to_csv('stacking_test.csv', index = False)
stacking_train_dataset.to_csv('stacking_train.csv', index = False)