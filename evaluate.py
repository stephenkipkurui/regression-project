# import feature engineering libraries
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Surpress warnings
import warnings
warnings.filterwarnings("ignore")


def select_kbest_feature_engineering(predictors, target, num_features):
    
    '''
        This function takes in predictors, and the target variables and the number of 
        features desired and returns the names of the top k selected features based on the SelectKBest class. 
    '''
    num_features = int(input('Enter count of SelectKBest features to return: '))
    
    kbest = SelectKBest(f_regression, k = num_features)
    
    kbest.fit(predictors, target)
    
    return predictors.columns[kbest.get_support()]


def rfe_feature_engineering(predictors, target, num_features):
    '''
        This function takes in predictors, and the target variables and the number of 
        features desired and returns the names of the top Recussion Feature Elimination(RFE) features 
        based on the SelectKBest class. 
    '''
    model = LinearRegression()
    
    num_features = int(input('Enter count of RFE features to return: '))
    
    rfe = RFE(model, n_features_to_select = num_features)
    
    rfe.fit(predictors, target)
    
    result = rfe.get_support()
    
    return predictors.columns[result]

def scaled_data(train, validate, test):
    
    '''
        This function takes in train, validate and test df and scales them then \'spits\' scaled df
    '''
    # call the scaler 
    scaler = StandardScaler()
    
    # Fit the scaler
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    return train_scaled, validate_scaled, test_scaled


def assessed_value_regression_plot(features, actual, predicted):
    
    plt.figure(figsize=(16, 10))
    plt.scatter(features, predicted, color='dimgray')
    
     # Plot regression line
    plt.plot(actual, features.yhat_predicted, color='darkseagreen', linewidth=3)

#     # add the residual line at y=0
#     plt.annotate('', xy=(70, 0), xytext=(100, 0), xycoords='data',
#                  textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})

    # set titles
    plt.title(r'Baseline Residuals', fontsize=12, color='black')
    # add axes labels
#     plt.ylabel(r'$\hat{y}-y$')
    plt.ylabel('Tipped Amount (USD)')

    plt.xlabel('Total Bill (USD')

    # add text
    plt.text(85, 15, r'', ha='left', va='center', color='black')

    return plt.show()