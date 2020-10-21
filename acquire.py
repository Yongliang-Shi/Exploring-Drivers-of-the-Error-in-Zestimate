import pandas as pd
import os
import env

#%%
def get_connection(db, user=env.user, host=env.host, password=env.password):
    """
    Returns the connection string to Codeup Database.
    Parameter: db name (str)
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# %%
def acquire_only(db,query):
    """
    Return df only according to the query. 
    Parameters: db(str.), querry(str.)
    """
    url = get_connection(db)
    df = pd.read_sql(query, url)
    return df

#%%
def get_zillow_data(query, iteration):
    """
    Returns in df format zillow dataset according to the querry.
    The df is read either directly from the local .csv or from the cloud.
    The df has the suffix indicating the version number.
    Parameters: querry(str), iteration(string int)
    """
    zillow_csv = 'zillow_' + iteration + '.csv'
    filename = zillow_csv
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        df = pd.read_sql(query, get_connection('zillow'))
        df.to_csv(filename)
        return df

def get_zillow_clustering():
    filename = 'zillow.csv'
    query = """
        select *
        from properties_2017
        join predictions_2017 using(parcelid)
        left join airconditioningtype using(airconditioningtypeid)
        left join architecturalstyletype using(architecturalstyletypeid)
        left join buildingclasstype using(buildingclasstypeid)
        left join heatingorsystemtype using(heatingorsystemtypeid)
        left join propertylandusetype using(propertylandusetypeid)
        left join storytype using(storytypeid)
        left join typeconstructiontype using(typeconstructiontypeid)
        where transactiondate between '2017-01-01' and '2017-12-31'
        and latitude is not null
        and longitude is not null
        order by parcelid, transactiondate
        """    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql(query, get_connection('zillow'))
        df.to_csv(filename)
        return df

# %%
def get_mall_data():
    """
    Returns mall dataset in df.
    """ 
    filename = 'mall_customers.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select * from customers""", get_connection('mall_customers'))
        df.to_csv(filename)
        return df

# %%
def get_iris_data():
    filename = 'iris.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else: 
        df = pd.read_sql("""select * from measurements join species using(species_id)""", get_connection('iris_db'))
        df.to_csv(filename)
        return df