import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %%
def obj_df(df):
    """
    Returns the dataframe only containing object columns
    Paramter: df
    """
    mask = np.array(df.dtypes == 'object')
    df_obj = df.iloc[:, mask]
    return df_obj

# %%
def sum_obj_cols (df):
    """
    Returns the object columns with their unique values (don't include NaN in counts). 
    Paramter: df
    """
    df_obj = obj_df(df)
    obj_cols = pd.DataFrame(df_obj.dtypes, columns=['dtypes'])
    obj_cols['unique_values'] = df_obj.nunique()
    return obj_cols

# %%
def obj_value_counts(df):
    """
    Count the unique values of each object column in the dataframe.
    Parameter: df
    """
    df_obj = obj_df(df)
    for col in df_obj.columns:
        print(df_obj[col].value_counts())
        print('-'*100)

# %%
def num_df(df):
    """
    Subset the dataframe only containing float or int columns
    Paramter: df
    """
    float_mask = np.array(df.dtypes == 'float')
    int_mask = np.array(df.dtypes == 'int')
    mask = float_mask | int_mask
    df_obj = df.iloc[:, mask]
    return df_obj

# %%
def sum_missing_values_attributes(df):
    """
    Count how many missing values in each attribute
    Parameter: interested df
    """
    missing_values = pd.DataFrame(df.isna().sum(axis=0), columns=['num_row_missing'])
    total_rows = df.shape[0]
    missing_values['pct_rows_missing'] = missing_values.num_row_missing/total_rows
    return missing_values

# %%
def sum_missing_values_cols(df):
    """
    Group the rows based on how many missing values they have
    Parameter: interested df
    """
    missing_values = df.isnull().sum(axis=1).value_counts().sort_index()
    cols_missing_values = {'num_cols_missing': missing_values.index.tolist(), 'num_rows': missing_values.values.tolist()}
    cols_missing_values = pd.DataFrame(cols_missing_values)
    total_rows = df.shape[0]
    cols_missing_values['pct_cols_missing'] = (cols_missing_values.num_rows/total_rows)*100
    return cols_missing_values

# %%
def plot_variable_pairs(df):
    """
    Pair-plot the variables
    Parameter: df
    """
    return sns.pairplot(df, kind='reg')