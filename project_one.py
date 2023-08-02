
## Predicting the GDP percapita for countries in year 2021 
## using sklearn linear regression model. 
## Author : Mehrnaz Sadroleslami 
## Date : Aug 2nd, 2023
## DataSet : https://www.kaggle.com/datasets/gwenaelmouthuy/gdp-per-capita-between-1960-and-2021

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("gdppercapita_us_inflation_adjusted.csv")

#get some info about the data 
#print (df.describe())

# let's clean the data first

#I want to know what is the datatype in the columns
# print (df.dtypes)
#Objects! first I need to change the data type to float/int types. 
def generate_float(value):
    if isinstance(value,str) and 'k' in value: 
        return float(value.replace('k','') )* 1000;
    elif pd.isnull(value):
        return value
    else: 
        return float(value)

for col in df.columns[1:]:
    df[col] = df[col].apply(generate_float)


#print (df.isnull().sum())

# now I will fill the NaNs with the mean 
target = df['2021']
target.dropna(inplace=True)

df_years = df.drop(columns=['country','2021'])
df_years = df_years.loc[target.index]
df_years.fillna(df_years.mean(), inplace=True)
#print (df_years.isnull().sum())


#split the data to the train and test
train_X,val_X,train_y,val_y = train_test_split(df_years,target, random_state=10)

model = LinearRegression()
model.fit(train_X,train_y)

prediction = model.predict(val_X)

mse = mean_squared_error(val_y,prediction)
r2 = r2_score(val_y,prediction)
print ("mean squared error with linear regression model : ",mse)
print ("R-Squared with the same model : ",r2)


## RESULT ## 
# mean squared error with linear regression model :  6815625.9060703525
# R-Squared with the same model :  0.9823244655561715

#mean squared error captures the difference between the predicted value and the 
# actual GDP for the year 2021. here we have a high mean squared error rate which means
# that the prediction is quiet far from the predictions. 
# value of r squared is a representation of how much the model is fitting the data and
# if it's capturing the general trend of data. 98% means that the model is explaining 
# most of the variance and trend in the data. 
# overally the fact that the mean squared error and r squared is high means that the data 
# is generally linear but for the year 2021 the trend changes or because of other factors 
# that we didn't consider in this model the prediction is not accurate. 