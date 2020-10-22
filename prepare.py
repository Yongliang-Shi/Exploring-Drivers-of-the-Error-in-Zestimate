import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

import acquire

# %%
def keep_last_transaction(zillow):
    zillow.drop_duplicates(subset=['parcelid'], keep='last', inplace=True, ignore_index=True)
    return zillow

# %%
def handle_missing_values(df, prop_required_column, prop_required_row):
    """
    Drop rows and columsn based on the perent of values that are missing.
    Parameters: 
    1. df
    2. the proportion, for each column, of rows with non-missing values requied to keep the column
    3. the proportion, for each row, of columns with non-missing values required to keep the row
    """
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# %%
def drop_remaining_missings(df):
    """
    Drop all the reamining missing values
    Parameter: df
    """
    mask = (df.isna().sum(axis=1) == 0)
    df = df[mask]
    return df

# %%
def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the series.
    The values returned will be either 0 (if the point is not an outlier), or a number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

# %%
def upper_outlier_columns(df, k):
    '''
    Return the column(s) with the upper_outliers for all the numeric columns in the given dataframe.
    '''
    df1 = pd.DataFrame()
    for col in df.select_dtypes('number'):
        df1[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df1

# %%
def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the series.
    The values returned will be either 0 (if the point is not an outlier), or a number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: min([x - lower_bound, 0]))

# %%
def lower_outlier_columns(df, k):
    '''
    Return the column(s) with the lower_outliers for all the numeric columns in the given dataframe.
    '''
    df1 = pd.DataFrame()
    for col in df.select_dtypes('number'):
        df1[col + '_lower_outliers'] = get_lower_outliers(df[col], k)
    return df1

# %%
def split_my_data(df, pct=0.10):
    '''
    This divides a dataframe into train, validate, and test sets. 
    Parameters - (df, pct=0.10)
    df = dataframe you wish to split
    pct = size of the test set, 1/2 of size of the validate set
    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(df, test_size=pct, random_state = 123)
    train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123)
    return train, validate, test

# %%
def split_stratify_my_data(df, strat, pct=0.10):
    '''
    This divides a dataframe into train, validate, and test sets straifying on the selected feature
    Parameters - (df, pct=0.10, stratify)
    df = dataframe you wish to split
    pct = size of the test set, 1/2 of size of the validate set
    stratify = a string of the column name of the feature you wish to stratify on
    Returns three dataframes (train, validate, test)
    '''
    train_validate, test = train_test_split(df, test_size=pct, random_state = 123, stratify=df[strat])
    train, validate = train_test_split(train_validate, test_size=pct*2, random_state = 123, stratify=train_validate[strat])
    return train, validate, test

# %%
def encode_label(df, columns_to_encode):
    """
    Dummy coding of categorical varibales
    Parameters: df, columns_to_encode(list)
    """
    obj_df = df[columns_to_encode]
    dummy_df = pd.get_dummies(obj_df, dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df.drop(columns=columns_to_encode, inplace=True)
    return df

# %%
def scale_minmax(train, validate, test, columns_to_scale):
    """
    Scale the columns using MinMaxScaler
    Return scaler object, train_scaled, validate_scaled, test_scaled
    Parameters: train(df), validate(df), test(df), columns_to_scale(list)
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    train.drop(columns=columns_to_scale, inplace=True)
    validate.drop(columns=columns_to_scale, inplace=True)
    test.drop(columns=columns_to_scale, inplace=True)
    return scaler, train, validate, test

# %%
def scale_standard(train, validate, test, columns_to_scale):
    """
    Scale the columns using StandardScaler
    Return scaler object, train_scaled, validate_scaled, test_scaled
    Parameters: train(df), validate(df), test(df), columns_to_scale(list)
    """
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler = StandardScaler()
    scaler.fit(train[columns_to_scale])
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    train.drop(columns=columns_to_scale, inplace=True)
    validate.drop(columns=columns_to_scale, inplace=True)
    test.drop(columns=columns_to_scale, inplace=True)
    return scaler, train, validate, test