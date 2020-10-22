import pandas as pd
import os
import env

#%%
def get_connection(db, user=env.user, host=env.host, password=env.password):
    """
    Returns the connection string to Codeup Database.
    Parameter: db: name of the database in string format
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# %%
def get_zillow_clustering():
    """
    Collect zillow dataset from the codeup clound database. 
    """
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