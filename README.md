README.md

# ZILLOW PROJECT 

## Project Goals and Description: 

Zillow Group, Inc is an American Online Real-estate Marketplace Company Housing Company founded in 2004 in Seattle, WA. With their online presense, Zillow Data Team has capitalized on capturing housing market data and every day millions and millions of data-points are captured, stored. This creates a need for dedicated team of scientists to explore and deduct meaning in these sets. This project is an example of on of the many studies on data to infer meaning. Our goal however, will be focused on the 2017 data for single household homes in three Carlifonia Counties- Orange, L.A. and Ventura. We will construct an additional new machine learning regression model to complement the current one in production with the ultimate purpose of predicting key drivers for assessed tax values for single family properties in the counties stated above. The approach here is adapting newer insight into keys driver features.

## Initial Hypothesis/Questions
Is there a correlation between assessed tax value and:- 

> - A home\'s square feet?
>- Home\'s number of bedrooms?
>- Home\'s number of bathrooms?
>- Transaction month?
>- Home\'s location?
>- Home's year built (age)?


## The Project Plan Structure 

This projects follows Data Science Pipeline. Should you wish to recreate the project, follow the following general steps:

# (i). Planning 
>- Define goals, understand the stakeholders and or audience, brain-storm to arrive at the MVP, define end product and purpose. Finally deliver report through presentation.


# (ii). Acquisition

>- Create a connection with CodeUp Online SQL through conn.py module.
>- Call the conn.py function with the acquire.py function.
>- Save a cached copy of the .csv file in local machine.
>- Import this module and other required python libraries into the main zillow_workspace.ipynb file

# (iii). Preparation

>- Create a module called prepare.py to hold all functions required to prepare the data for exploration including: 
     - Remove duplicates, empty spaces, encode data, drop unnecessary columns and data outliers, rename columns. 
     - Split the data into train, validate and test sets in ratios: 56% : 24% : 20% respectively.


# (iv). Exploration

>- With cleaned data prepared from preparion phase, ensure no data leakage and use train subset.
>- Utelize the initial questions to guide the project explorations.
>- Create visualizations to explain the relationships observed.
>- Perform statistical test to infer key driveres through feature engineering.
>- Document takeaways that guide the project to modeling.

# (v). Modeling
>- Utilize python Sklearn libraries to create, fit and test models in the zillow workspace.ipynb file
>- Predict the target variables 
>- Evaluate the results on the in-sample predictions
>- Select the best model, and use on test subsets for out-of-sample observations.

# (vi). Delivery

>- A final report with summarized results is saved in the zillow_report workbook.
>- Deliverables include a 5 minute presentation through Zoom WebCast with the audience of Zillow Data Science Team. 
>- The key drivers for asssessed tax values stated clearly and best perfoming model presented backed by figures and visul charts. 
>- Deployment of the entire code and workbooks in public Data Scientist GitHub Account with strict exclusion of sensitive database access information through .gitignore.py. 
>- Create this** ReadMe.md file with all steps required to reproduce this test.

# Appendix

Data Dictionary 

|Column | Description | Dtype|
|--------- | --------- | ----------- |
|bed_count | The number of bedrooms | float64 |
|bath_count | The number of bathrooms | float64 |
|square_feet | Square footage of property | float64 |
|assessed_value | Property tax value dollar amount | float64 |
|built_year | Year the property was built | int64 |
|zip_code | Federal Information Processing Standard code | int64 |
|city | City property located | str |
|home_age | The age of home till 2017 | int64
|trans_day | The day of month home transactioned | int64 |

# Steps to Reproduce this Project

>- You'll require own env.py file to store query connection credentials(access rights to CodeUp SQL is required).
>- Read this ReadMe.md file fully and make necessary files
>- Set up .gitignore file and ignore env.py, *.csv files
>- Create and Repository directory on GitHub.com
>- Follow instructions above and run the final zillow_workspace report.


# Model


# Key Findings 

# Future Explorations of Interest


