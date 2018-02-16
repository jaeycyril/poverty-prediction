import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def stan(df):
    '''
    Function to standardze dataframe. 
    '''
    numeric = df.select_dtypes(include=["float64", "int64"])
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df


def pre_process_data(df, enforce_cols=None):
    print("input shape\t{}".format(df.shape))
    
    # standardize numerical columns of dataframe
    df = stan(df)
    print("After standardization\t{}".format(df.shape))
    
    # get one hot encoding for categorical variables
    df = pd.get_dummies(df)
    print("After one hot encoding of categoricals\t{}".format(df.shape))
    
    # match training and test set
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)
        
        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    # fill all Nan values in dataset with zeros
    #df.fillna(0, inplace=True)
    
    return df


def impute_vals(df):
    for column in df.columns:
        if(is_numeric_dtype(df[column])):
            df[column].fillna(df[column].mean(), inplace=True)

    return df