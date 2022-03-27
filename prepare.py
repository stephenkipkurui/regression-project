import acquire
from acquire import get_zillow
import pandas as pd
import numpy as np
import datetime

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prepare_zillow():
    
    # Acquire zilloW df from acquire module
    zillow = get_zillow()
        
    # Renaming cols for readability
    zillow = zillow.rename(columns = ({'bedroomcnt':'bed_count','bathroomcnt':'bath_count',
                                     'calculatedfinishedsquarefeet':'square_feet',
                                       'taxvaluedollarcnt':'assessed_value', 'fips':'fips',
                                       'yearbuilt':'year_built','transactiondate':'trans_date'}))
                           
    # Drop nulls                       
    zillow = zillow.dropna()
    
    # Drop duplicate columns
    zillow = zillow.drop_duplicates()

    # Reset index
    zillow = zillow.reset_index(drop = True)
    
    zillow['trans_date'] = pd.to_datetime(zillow['trans_date'], format = '%Y-%m-%d')
    
    zillow['trans_month'] = pd.to_datetime(zillow['trans_date']).dt.month

# #     zillow['trans_month'] = pd.DatetimeIndex(zillow['trans_date']).month
# #     zillow['trans_day'] = pd.DatetimeIndex(zillow['trans_date']).day

#     zillow['trans_month'] = zillow['trans_date'].dt.month
#     zillow['trans_day']   = zillow['trans_date'].dt.day

    
#     zillow[['month', 'date', 'year']] = zillow['trans_date'].str.split('-', expand = True)

#     # Strip white spaces if any
#     zillow = zillow.strip()
    
    
#     # Strip zeros from year
#     zillow = zillow.year_built.str.rstrip('.0')
    
    # Split the data into train, validate and test
    train_validate, test = train_test_split(zillow, test_size=0.2, 
                                                random_state=123)
    
    train, validate = train_test_split(train_validate,
                                        test_size=0.3,
                                       random_state=123)

    # Function return
    return train, validate, test
    



  
    
  
    
    