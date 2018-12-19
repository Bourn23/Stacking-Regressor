import pandas as pd
"""
Remember:
0. how we picked from splitted data using iloc

"""
df_train1 = pd.read_csv('001_train.csv')
predicators = ['Item_Weight', 'Item_Fat_Content',
       'Item_MRP', 'Item_Visibility', 'Items_ID',
        'Outlet_Size','Outlet_Age', 'Outlet_Location_Type', 'Outlet_Type' ]
target = df_train1['Item_Outlet_Sales']
df_train = df_train1.drop('Item_Outlet_Sales', axis = 1)
df_train = df_train[predicators]

df_test = pd.read_csv('001_test.csv')


# print(df_train.head())
# print('-------------TEST-----------')
# print(df_test.head())
import pandas as pd
import numpy as np
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

base_models = [DecisionTreeRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, XGBRegressor]

# stacking_train_dataset = np.matrix([[row for row in range(len(df_train))],[col for col in range(len(base_models))]])
row_train = len(df_train)
col_train = len(base_models)
stacking_train_dataset1 = np.zeros(shape = (row_train, col_train))
row_test = len(df_test)
col_test = len(base_models)
stacking_test_dataset1 = np.zeros(shape = (row_test, col_test))
stacking_train_dataset = pd.DataFrame(data= stacking_train_dataset1)
stacking_test_dataset = pd.DataFrame(data = stacking_test_dataset1)
# def model_combiner (base_models, )
split = KFold(n_splits=10, random_state=42)
# print(len(df_train.columns))
for i, base_alg in enumerate(base_models):
    for trainix , testix in split.split(stacking_train_dataset):
        x_train = df_train.iloc[trainix]
        y_train = target.iloc[trainix] #REMEMBER HOW WE CHOOSE ILOC
        x_test = df_train.iloc[testix]
        y_test = df_train.iloc[testix]
        model = base_alg()
        print(type(model))
        # stacking_train_dataset['testcv', i] = 
        # base_alg.fit(X, Y)#.predict(df_test[trainix])
        model.fit(x_train, y_train)
    #     base_alg.fit(df_train[rownum], target[row_num])
    #     for row_num_testix in testix:
        # print(type(stacking_train_dataset))

        stacking_train_dataset['Item_Outlet_Sales', i] = model.predict(x_train)
    for j in range(len(stacking_test_dataset)):
        stacking_test_dataset [j, i] = model.fit(x_test, y_test).predict(df_test)

ensemble_model = LinearRegression()

# for i in range(len(stacking_test_dataset)):


df_predict_final = ensemble_model.fit(stacking_train_dataset, target).predict(stacking_test_dataset)
print(type(df_predict_final))

df_predict = pd.DataFrame(data = pd_predict_final)
df_stack_test.to_csv('stacking_test.csv', index = False)
df_stack_train.to_csv('stacking_train.csv', index = False)
df_predict_final.to_csv('stacking_predict.csv', index = False)





