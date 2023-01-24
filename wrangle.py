import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from env import get_db_url

from sklearn.model_selection import train_test_split
import sklearn.preprocessing


def get_data(sql_db, query):
    '''
        Accepts 2 arguments of string type:
        1: SQL database name
        2: SQL query
        
        Checks if .csv already exists before
        connecting with SQL database again
        
        Saves a .csv file of DataFrame
        
        Returns DataFrame
    '''
    
    import os
    import pandas as pd
    
    # variable to hold filename created from 
    # input argument of SQL database name
    path = f'{sql_db}.csv'
    
    # Holds boolean result of check for
    # .csv existing; uses OS module
    file_exists = os.path.exists(path)
    
    # Uses boolean value variable to
    # check whether to create a new
    # SQL connection or load .csv
    #
    # Finished off by returning DataFrame
    if file_exists:
        print('Reading CSV')
        df = pd.read_csv(path)
        
        return df

    else:
        print('Downloading SQL DB')

        url = get_db_url(sql_db)
        df = pd.read_sql(query, url)
        df.to_csv(f'{sql_db}.csv',index=False)
        return df

def clean_data(df):
    '''
        Accepts DataFrame from get_data() function in wrangle.py
            &
        Returns a cleaned DataFrame
    '''
    
    # Drop Null Columns
    # =================

    # List of Null Column Values
    df.isnull().sum().sort_values(ascending=False)[:34]

    # List of Null Column Names
    drop_column_list = ["buildingclasstypeid"\
                        , "finishedsquarefeet15"\
                        , "finishedsquarefeet13"\
                        , "storytypeid"\
                        , "basementsqft"\
                        , "yardbuildingsqft26"\
                        , "architecturalstyletypeid"\
                        , "typeconstructiontypeid"\
                        , "fireplaceflag"\
                        , "finishedsquarefeet6"\
                        , "decktypeid"\
                        , "pooltypeid10"\
                        , "poolsizesum"\
                        , "pooltypeid2"\
                        , "hashottuborspa"\
                        , "yardbuildingsqft17"\
                        , "taxdelinquencyflag"\
                        , "taxdelinquencyyear"\
                        , "finishedfloor1squarefeet"\
                        , "finishedsquarefeet50"\
                        , "threequarterbathnbr"\
                        , "fireplacecnt"\
                        , "pooltypeid7"\
                        , "poolcnt"\
                        , "airconditioningtypeid"\
                        , "numberofstories"\
                        , "garagetotalsqft"\
                        , "garagecarcnt"\
                        , "regionidneighborhood"\
                        , "buildingqualitytypeid"\
                        , "unitcnt"\
                        , "propertyzoningdesc"\
                        , "heatingorsystemtypeid"]
                        
    # Dropping Columns using 'drop_column_list'
    df.drop(columns = drop_column_list, inplace=True)


    # Drop Null Rows
    # ==============

    # List of Null Rows Values
    less_than_1100_nulls = df.isnull().sum().sort_values(ascending=False)[:13]

    # List of Null Rows Names
    less_than_1100_nulls_list = ['regionidcity'\
                            ,'lotsizesquarefeet'\
                            ,'finishedsquarefeet12'\
                           ,'calculatedbathnbr'\
                           ,'fullbathcnt'\
                           ,'censustractandblock'\
                           ,'yearbuilt'\
                           ,'structuretaxvaluedollarcnt'\
                           ,'calculatedfinishedsquarefeet'\
                           ,'regionidzip'\
                           ,'taxamount'\
                           ,'landtaxvaluedollarcnt'\
                           ,'taxvaluedollarcnt']
                           
    # Drop Null Rows using subset and list of names
    df = df.dropna(subset=less_than_1100_nulls_list)


    #Fix Data Types
    # ==============

    # temporarily converting to interger to remove
    # trailing zeroes
    df['fips'] = df['fips'].apply(int).copy()

    # converting to final datatype as string
    df['fips'] = df['fips'].apply(str).copy()

    # as string adding a leading '0'
    df['fips'] = '0' + df['fips'].copy()

    # convert 'yearbuilt' to interger
    df['yearbuilt'] = df['yearbuilt'].apply(int).copy()

    # convert 'assessmentyear' to interger
    df['assessmentyear'] = df['assessmentyear'].apply(int).copy()


    # Rename Columns
    # ==============

    # list of new names
    new_column_names = ['parcelid'\
    ,'id'\
    ,'bathroom_count'\
    ,'bedroom_count'\
    ,'calculated_bathandbr'\
    ,'calculated_finished_square_feet'\
    ,'finished_square_feet_12'\
    ,'fips'\
    ,'full_bath_count'\
    ,'latitude'\
    ,'longitude'\
    ,'lot_size_square_feet'\
    ,'property_county_landuse_code'\
    ,'property_land_use_type_id'\
    ,'raw_census_tract_and_block'\
    ,'region_id_city'\
    ,'region_id_county'\
    ,'region_id_zip'\
    ,'room_count'\
    ,'year_built'\
    ,'structure_taxvalue_dollarcount'\
    ,'tax_valuedollar_count'\
    ,'assessment_year'\
    ,'land_tax_value_dollar_count'\
    ,'tax_amount'\
    ,'census_tract_and_block'\
    ,'id_1'\
    ,'log_error'\
    ,'transaction_date']

    # renaming
    df.set_axis(new_column_names, axis=1,inplace=True)

    # feature engineering
    df.drop(columns='calculated_bathandbr',inplace=True)
    df['bed_bath_count']=df['bathroom_count']+df['bedroom_count']
    
    return df

def split_data(df,target=None,seed=42):    
    '''
        Accepts 2 arguments;
        DataFrame, and Keyword Value for 'stratify'
        ...
        Splits DataFrame into a train, validate, and test set
        and it will return three DataFrames stratified on target:
        
        train, val, test (in this order) -- all pandas Dataframes
        60%,20%,10% 
    '''
    
    train, test = train_test_split(df,
                               train_size = 0.8, random_state=seed)
    train, val = train_test_split(df,
                             train_size = 0.75,
                             random_state=seed)
    return train, val, test

def model_data(df,target=None):
    '''
        Isolates Target Variable from DataFrame
        Returns X & y DataFrames
    '''
    X = df.drop(columns=target)
    y = df[target]
    return X,y

def wrangle_zillow():
    '''
        Main function in `wrangle.py`
        When run, wrangle_zillow will utilize
        get_db_url(), get_data(), and clean_data()
        
        to acquire & prepare DataFrame
        
        returns a DataFrame
    '''
    
    sql_db = "zillow"
    query = "SELECT * FROM properties_2017 JOIN predictions_2017 USING(parcelid) WHERE (`propertylandusetypeid` = 261) & (YEAR(`transactiondate`) = 2017);"
    df = get_data(sql_db,query)
    df = clean_data(df)
    
    return df

def MinMaxScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Min Max Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def StandardScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Standard Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def RobustScaler(df):
    '''
        Accepts a DataFrame
        
        Creates a Robust Scaler object
        Fits & Transforms DataFrame
        
        Returns scaled DataFrame
    '''
    
    scaler = sklearn.preprocessing.RobustScaler()
    scaler.fit(df)

    df_scaled = scaler.transform(df)
    
    return df_scaled

def compare_scalers(df):
    '''
        Accepts a DataFrame
        
        Is used to visualize 3 scaler outputs
        and compare to original DataFrame
    '''
    mm_scaled = MinMaxScaler(df)
    ss_scaled = StandardScaler(df)
    rs_scaled = RobustScaler(df)

    font = {'family': 'Georgia',
            'color':  '#525252',
            'weight': 'bold',
            'size': 25,
            }
    # ====================================================================

    # Assigning 'fig', 'ax' variables.
    fig, ax = plt.subplots(2, 2,figsize=(25,25))

    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (0, 8000)

    # Setting the values for all axes.
    #plt.setp(ax, xlim=custom_xlim)
    
    # ====================================================================
    
    # Original Data
    ax[0][0].hist(df, color="#525252",ec='white',bins=10000)
    ax[0][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][0].set_title("Original",color='#525252', fontdict=font)
    ax[0][0].set_xlim([0, 8000])
    
    
    # MinMax Scaled
    ax[0][1].hist(mm_scaled, color="#525252",ec='white',bins=10000)
    ax[0][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[0][1].set_title("MinMax Scaled",color='#525252', fontdict=font)
    ax[0][1].set_xlim([0, .005])
    
    # Standard Scaled
    ax[1][0].hist(ss_scaled, color="#525252",ec='white',bins=10000)
    ax[1][0].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][0].set_title("Standard Scaled",color='#525252', fontdict=font)
    ax[1][0].set_xlim([-1.5, 3])

    # Robust Scaled
    ax[1][1].hist(rs_scaled, color="#525252",ec='white',bins=10000)
    ax[1][1].set_ylabel(r"y", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_xlabel(r"x", fontsize=14, color="#525252", fontdict=font)
    ax[1][1].set_title("Robust Scaled",color='#525252', fontdict=font)
    ax[1][1].set_xlim([-2, 5])

def get_rmse(value,pred):
    '''
    rmse with actual values and predicted values
    '''
    return mean_squared_error(value,pred)**(1/2)


def rfe(X,y,k):

    olm = LinearRegression()
    rfe = RFE(olm,n_features_to_select=k)
    rfe.fit(X,y)
    
    mask = rfe.support_
    
    return X.columns[mask]

def select_kbest(X,y,k):

    f_selector = SelectKBest(f_regression,k=k)
    f_selector.fit(X,y)
    mask = f_selector.get_support()
    return X.columns[mask]
def eval_results(p, alpha, group1, group2):
    '''
    this function will take in the p-value, alpha, and a name for the 2 variables
    you are comparing (group1 and group2) and return a string stating 
    whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        return f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'
    else:
        return f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'