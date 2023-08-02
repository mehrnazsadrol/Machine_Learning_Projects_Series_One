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


print (df.isnull().sum())
# now I will fill the NaNs with the mean 
df_years = df.drop(columns='country')
df_clean = df.fillna(df_years.mean(), inplace=True)
print (df.isnull().sum())
# print (df.head(100))
#split the data to the train 

# print ( df_clean.head()); 