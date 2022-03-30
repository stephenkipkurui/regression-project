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
#------------------


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


def baseline_vs_model_regression(df, x, y, yhat):
    
    plt.figure(figsize=(16, 9))

    ## plot data points, regression line and baseline
    # plot the data points
    plt.scatter(x, y, color='dimgray', s=40)

    # plot the regression line
    plt.plot(x, yhat, color='darkseagreen', linewidth=3)

    # add baseline through annotation
    plt.annotate('', xy=(70, y.mean()), xytext=(102, y.mean()), xycoords='data', textcoords='data', 
                 arrowprops={'arrowstyle': '-', 'color': 'goldenrod', 'linewidth': 2, 'alpha': .75})

    ## ---------------------------------------------
    ## add line labels
    # the regression line equation
    plt.text(89.5, 90.5, r'$\hat{y}=12.5 + .85x$',
             {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center', 'rotation': 27})

    # the baseline equation
    plt.text(88, 82, r'$\hat{y}=83$',
             {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center'})

    ## ---------------------------------------------
    # set and plot title, subtitle, and axis labels
    # set titles
    title_string = r'Difference in Error'
    subtitle_string = "Baseline vs. Regression Line"

    # add titles
    plt.title(subtitle_string, fontsize=12, color='black')
    plt.suptitle(title_string, y=1, fontsize=14, color='black')

    # add axes labels
    plt.ylabel('Assessed Value')
    plt.xlabel('Home Square Feet')

    ## ----------------------------------------
    # annotate each data point with an error line to the baseline and the error value
    for i in range(len(df)):

        # add error lines from baseline to data points
        plt.annotate('', xy=(x[i]+.1, y[i]), xytext=(x[i]+.1, y.mean()), xycoords='data', textcoords='data',
                     arrowprops={'arrowstyle': '-', 'color': 'goldenrod', 'linestyle': '--', 'linewidth': 2, 'alpha': .5})
        # add error lines from regression line to data points
        plt.annotate('', xy=(x[i], y[i]), xytext=(x[i], yhat[i]), xycoords='data', textcoords='data',
                     arrowprops={'arrowstyle': '-', 'color': 'darkseagreen', 'linestyle': '--', 'linewidth': 2, 'alpha': .75})

    ## ----------------------------------------
    # annotate some of the error lines with pointers
    # add pointer: the first data point to the regression line
    plt.annotate('', xy=(70.25, 70), xytext=(73, 70), xycoords='data', textcoords='data',
                 arrowprops={'arrowstyle': 'fancy', 'color': 'darkseagreen', 'linewidth': 1})

    # add pointer: the last data point to the regression line
    plt.annotate('', xy=(100.25, 97), xytext=(103, 97), xycoords='data', textcoords='data',
                 arrowprops={'arrowstyle': 'fancy', 'color': 'darkseagreen', 'linewidth': 1})

    # add pointer: the last data point to the baseline
    plt.annotate('', xy=(100.25, 90), xytext=(103, 90), xycoords='data', textcoords='data',
                 arrowprops={'arrowstyle': 'fancy', 'color': 'goldenrod', 'linewidth': 1})

    ## ----------------------------------------
    ## add text to the annotatations
    # the error of the first data point to the regression line
    plt.text(73, 70, 4.1, ha='left', va='center', color='black')

    # the error of the last data point to the regression line
    plt.text(103, 96, 1.6, ha='left', va='center', color='black')

    # the error of the last data point to the baseline
    plt.text(103, 90, -12.7, ha='left', va='center', color='black')

    ## ----------------------------------------

    return plt.show()