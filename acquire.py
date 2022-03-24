import pandas as pd
import env
import os

def db_conn():
    
    db = 'zillow'
    
    url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'
        
    return url


def get_zillow(use_cache = True):
    
    zillow_file = 'zillow.csv'
    
    if os.path.exists(zillow_file) and use_cache:
        
        print('Status: Acquiring data from cached csv file..')
        
        return pd.read_csv(zillow_file)
    
    qry = '''
         SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, 
                 yearbuilt, fips, pred17.transactiondate
                 
         FROM properties_2017 
         
         JOIN propertylandusetype plt USING (propertylandusetypeid)
         
         JOIN predictions_2017 pred17 USING (parcelid)
         
         WHERE plt.propertylandusedesc = 'Single Family Residential';
    
          '''
    
    print('Status: Acquiring data from SQL database..')
    
    zillow = pd.read_sql(qry, db_conn())
    
    print('Status: Saving zillow data locally..')
    
    zillow.to_csv(zillow_file, index = False)
    
    
    