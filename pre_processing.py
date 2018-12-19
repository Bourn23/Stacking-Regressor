import pandas as pd
import category_encoders as ce
from fancyimpute import KNN


def pre_processor(training_data):

    # df = training_data
    # pd.set_option('display.max_columns', 50)
    # df = pd.read_csv("data/Train_UWu5bXk.csv")
    # training_data = df
    # print(df.info())


    # ############################ Encoding Section ########################

    # -----------'Item_Identifier'----------------
    # print(training_data[['Item_Identifier', 'Item_Outlet_Sales']].corr())
    encoder = ce.BinaryEncoder(cols=['Item_Identifier'])
    training_data = encoder.fit_transform(training_data)
    # print(training_data.columns)

    # -----------'Item_Weight'----------------
    # print(training_data['Item_Weight'].value_counts())


    # -----------'Item_Fat_Content'---------------
    training_data['Item_Fat_Content'] = training_data['Item_Fat_Content'].astype('category').replace({'Low Fat': 0,
                                                                                'low fat': 0,
                                                                                'LF': 0,
                                                                                'Regular': 1,
                                                                                'reg': 1})


    # -----------'Item_Visibility'---------------
    # print(training_data['Item_Visibility'].value_counts())


    # -----------'Item_Type'---------------
    encoder = ce.BinaryEncoder(cols=['Item_Type'])
    training_data = encoder.fit_transform(training_data)


    # -----------'Item_MRP'---------------
    # print(training_data['Item_MRP'].value_counts())


    # -----------'Outlet_Identifier'---------------
    encoder = ce.BinaryEncoder(cols=['Outlet_Identifier'])
    training_data = encoder.fit_transform(training_data)


    # -----------'Outlet_Establishment_Year'---------------
    # print(training_data['Outlet_Establishment_Year'].value_counts())


    # -----------'Outlet_Size'---------------
    # print(training_data['Outlet_Size'].value_counts())
    training_data['Outlet_Size'] = training_data['Outlet_Size'].astype('category').replace({'Small': 0, 'Medium': 1, 'High': 2})
    training_data['Outlet_Size'] = training_data['Outlet_Size'].astype('int64', errors='ignore')
    # print(type(training_data))
    # print("---------")
    # print(type(training_data['Outlet_Size']))
    # print("---------")
    # print(type(training_data['Outlet_Size'][10]))
    # print(training_data['Outlet_Size'].value_counts())
    # print(training_data.info())
    # print(type(training_data['Outlet_Size'][0]))


    # -----------'Outlet_Location_Type'---------------
    # print(training_data['Outlet_Location_Type'].value_counts())
    training_data['Outlet_Location_Type'] = training_data['Outlet_Location_Type'].replace({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})


    # -----------'Outlet_Type'---------------
    # print(training_data['Outlet_Type'].value_counts())
    encoder = ce.BinaryEncoder(cols=['Outlet_Type'])
    training_data = encoder.fit_transform(training_data)


    # -----------'Item_Outlet_Sales'---------------
    # print(training_data['Item_Outlet_Sales'].value_counts())


    # print(training_data.info())


    # ####################### Impute Section #######################

    data = KNN(k=5).fit_transform(training_data)
    df_modified = pd.DataFrame(data=data)

    return df_modified


