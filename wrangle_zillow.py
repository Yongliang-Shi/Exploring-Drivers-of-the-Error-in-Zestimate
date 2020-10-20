import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import prepare

# %%
def wrangle_zillow_mvp(zillow):
    """
    Wrangle zillow dataset: 
    1. keep the last transaction for each property
    2. keep the single unit properies only (260, 261, 262, 279)
    3. keep the prop_required_column = 0.6 and prop_required_rwo = 0.75
    4. Drop all the remaining_missings
    """
    zillow = prepare.keep_last_transaction(zillow)
    single_unit = [260, 261, 262, 279]
    zillow = zillow[zillow.propertylandusetypeid.isin(single_unit)]
    zillow = prepare.handle_missing_values(zillow, 0.6, 0.75)
    zillow = prepare.drop_remaining_missings(zillow)
    return zillow
# %%
def wrangle_zillow_clustering(zillow):
    """
    Wrangle zillow dataset: 
    1. keep the last transaction for each property
    2. keep the single unit properies only (260, 261, 262, 279)
    3. keep the prop_required_column = 0.6 and prop_required_rwo = 0.75
    4. Drop all the remaining_missings
    """
    zillow = prepare.keep_last_transaction(zillow)
    single_unit = [260, 261, 262, 279]
    zillow = zillow[zillow.propertylandusetypeid.isin(single_unit)]
    zillow = zillow[(zillow.bedroomcnt > 0) & (zillow.bathroomcnt > 0)]
    zillow.unitcnt = zillow.unitcnt.fillna(1.0)
    zillow = zillow[zillow.unitcnt == 1.0]
    zillow = prepare.handle_missing_values(zillow, 0.6, 0.60)
    zillow = zillow.drop(columns=['id', 'id.1'])
    zillow.drop(columns=['propertyzoningdesc'], inplace=True)
    zillow.drop(columns=['heatingorsystemtypeid'], inplace=True)
    zillow.heatingorsystemdesc = zillow.heatingorsystemdesc.fillna("None")
    zillow.drop(columns=["calculatedbathnbr"], inplace=True)
    train, validate, test = prepare.split_my_data(zillow, pct=0.15)
    cols = [
            "buildingqualitytypeid",
            "regionidcity",
            "regionidzip",
            "yearbuilt",
            "regionidcity",
            "censustractandblock"
            ]     
    for col in cols:
        mode = int(train[col].mode()) 
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value=mode, inplace=True)
        test[col].fillna(value=mode, inplace=True)
    cols = [
            "structuretaxvaluedollarcnt",
            "taxamount",
            "taxvaluedollarcnt",
            "landtaxvaluedollarcnt",
            "structuretaxvaluedollarcnt",
            "finishedsquarefeet12",
            "calculatedfinishedsquarefeet",
            "fullbathcnt",
            "lotsizesquarefeet"
            ]
    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)
    return train, validate, test
# %%
def wrangle_zillow_clustering_2(zillow):
    """
    Wrangle zillow dataset: 
    1. keep the last transaction for each property
    2. keep the single unit properies only (260, 261, 262, 279)
    3. keep the prop_required_column = 0.6 and prop_required_rwo = 0.75
    4. Drop all the remaining_missings
    """
    zillow = prepare.keep_last_transaction(zillow)
    single_unit = [260, 261, 262, 279]
    zillow = zillow[zillow.propertylandusetypeid.isin(single_unit)]
    zillow = zillow[(zillow.bedroomcnt > 0) & (zillow.bathroomcnt > 0)]
    zillow.unitcnt = zillow.unitcnt.fillna(1.0)
    zillow = zillow[zillow.unitcnt == 1.0]
    zillow = prepare.handle_missing_values(zillow, 0.6, 0.60)
    zillow = zillow.drop(columns=['id', 'id.1'])
    zillow.drop(columns=['propertyzoningdesc'], inplace=True)
    zillow.drop(columns=['heatingorsystemtypeid', 'propertylandusetypeid', 'unitcnt', 
                         'assessmentyear', 'transactiondate', 'propertylandusedesc', 
                         'regionidcounty', 'regionidcity', 'regionidzip', 
                         'rawcensustractandblock', 'censustractandblock', 'propertycountylandusecode'], inplace=True)
    zillow.heatingorsystemdesc = zillow.heatingorsystemdesc.fillna("None")
    zillow.drop(columns=["calculatedbathnbr"], inplace=True)
    train, validate, test = prepare.split_my_data(zillow, pct=0.15)
    cols = ["buildingqualitytypeid", "yearbuilt"]     
    for col in cols:
        mode = int(train[col].mode()) 
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value=mode, inplace=True)
        test[col].fillna(value=mode, inplace=True)
    cols = [
            "structuretaxvaluedollarcnt",
            "taxamount",
            "taxvaluedollarcnt",
            "landtaxvaluedollarcnt",
            "finishedsquarefeet12",
            "calculatedfinishedsquarefeet",
            "fullbathcnt",
            "lotsizesquarefeet"
            ]
    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)
    return train, validate, test
