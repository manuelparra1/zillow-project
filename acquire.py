import pandas as pd
import numpy as np

import os
from env import get_db_url

def get_data(sql_db, query):
    """
        Accepts 2 arguments of string type:
        1: SQL database name
        2: SQL query
        
        Checks if .csv already exists before
        connecting with SQL database again
        
        Saves a .csv file of DataFrame
        
        Returns DataFrame
    """
    
    import os
    import pandas as pd
    
    # variable to hold filename created from 
    # input argument of SQL database name
    path = f"{sql_db}.csv"

    # Holds boolean result of check for
    # .csv existing; uses OS module
    file_exists = os.path.exists(path)
    
    # Uses boolean value variable to
    # check whether to create a new
    # SQL connection or load .csv
    #
    # Finished off by returning DataFrame
    if file_exists:
        df = pd.read_csv(path)
        
        print("Reading CSV")
        return df

    else:
        url = get_db_url(sql_db)
        df = pd.read_sql(query, url)
        df.to_csv(f"{sql_db}.csv")
        
        print('Downloading SQL DB')
        return df
def clean_data(df):
    '''
        Accepts DataFrame from get_data() function in wrangle.py
            &
        Returns a cleaned DataFrame
    '''
    
    meat_potatoes

    return df
def wrangle_zillow():
    '''
        Main function in `wrangle.py`
        When run, wrangle_zillow will utilize:
        
        get_db_url(), get_data(), and clean_data()
        
        to acquire & prepare DataFrame
        
        returns a DataFrame
    '''
    sql_db = "zillow"
    query = "SELECT * FROM properties_2017 JOIN predictions_2017 USING(parcelid) WHERE (`propertylandusetypeid` = 261) & (YEAR(`transactiondate`) = 2017);"
    df = get_data(sql_db,query)
    #df = clean_data(df)
    
    return df
