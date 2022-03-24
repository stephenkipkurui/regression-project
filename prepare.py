import acquire
from acquire import get_zillow
import pandas as pd
import numpy as np

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
                                       'taxvaluedollarcnt':'assessed_value',
                                       'yearbuilt':'year_built','transactiondate':'trans_date'}))
                           
    # Drop nulls                       
    zillow = zillow.dropna()
    
    # Drop duplicate columns
    zillow = zillow.drop_duplicates()

    # Reset index
    zillow = zillow.reset_index(drop = True)
    
    # Split the data into train, validate and test
    train_validate, test = train_test_split(zillow, test_size=0.2, 
                                                random_state=123)
    
    train, validate = train_test_split(train_validate,
                                        test_size=0.3,
                                       random_state=123)
    # Convert to pandas DataFrame
    train = pd.DataFrame(train, columns=['bed_count','bath_count','square_feet',
                                                          'assessed_value','year_built','trans_date'])
    validate = pd.DataFrame(validate, columns=['bed_count','bath_count','square_feet',
                                                          'assessed_value','year_built','trans_date'])
    test = pd.DataFrame(test, columns=['bed_count','bath_count','square_feet',
                                                          'assessed_value','year_built','trans_date'])

    # Function return
    return train, validate, test
    


   
  
    
  
    
    