# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:46:15 2022

@author: Jan Chojnacki
"""
import pandas as pd
import numpy as np

path = "amex-default-prediction"

df = pd.read_csv(f"{path}"+"/test_data.csv",nrows=1000)

#memory cost reduction
# inspired by https://www.kaggle.com/competitions/amex-default-prediction/discussion/328054
# we could save customer_ID as int 64 through hexadecimal string encoding :
    
#df.customer_ID.map(lambda x: int(x[-16:],16)).astype("int64")

#Or assign an int32 index to all customers' ID 
ID_to_index_dict = dict(zip(df.customer_ID, df.index))
#When it comes to the submission we'll need to decode our indices:
index_to_ID_dict = dict(zip(df.index, df.customer_ID))

df.customer_ID = df.customer_ID.map(ID_to_index_dict).astype("int64")

#We also can lessen the date size by using int8 and creating 3 columns with year, month, day
df["two_digits_year"] = df.S_2.map(lambda x: x[2:4]).astype("int8")
df["month"] = df.S_2.map(lambda x: x[-5:-3]).astype("int8")
df["day"] = df.S_2.map(lambda x: x[-2:]).astype("int8")

df.drop("S_2",axis=1,inplace=True)

from sklearn.preprocessing import OrdinalEncoder 

#Reduce categorical columns to int8 (maximal 8 different entries) and drop rows which contain a NaN value
categorical_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
categorical_df = df[categorical_columns].copy()
categorical_df.dropna(inplace=True)

shaved_index = categorical_df.index
to_shave_index = set(df.index)-set(shaved_index)

encoder = OrdinalEncoder()
ordinal_encoded_df = pd.DataFrame(encoder.fit_transform(categorical_df),dtype="int8")

ordinal_encoded_df.columns = categorical_columns
ordinal_encoded_df.index = shaved_index

df.drop(to_shave_index, inplace=True)
df.update(ordinal_encoded_df)
df[categorical_columns] = df[categorical_columns].astype("int8")

