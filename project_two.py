
## Analyzing the failed bank dataset using different statistical and visualization methods.
## Author : Mehrnaz Sadroleslami 
## Date : Aug 3rd, 2023
## DataSet : https://www.kaggle.com/datasets/neutrino404/failed-banks-fdic-data

import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv("Failed_Bank_Dataset_2.csv")


#cleaning data 
df_clean = df['Estimated_Loss']
df_clean.dropna(inplace=True)
df = df.loc[df_clean.index]
#print ( df.isnull().sum())

#now that the data is clean. I want to get some basic information about it! 
# print (df.describe())
# print ("-----------------")
# print (df.columns)
# print ("-----------------")
# print (df.dtypes)
# print (df.shape)

# let's see if there is a correlation between the total assets that a bank owned and the 
#estimated loss of this bank 
data_frame = df.copy()
data_frame.drop(['Bank_Name', 'City', 'State', 'Cert', 'Acquiring_Institution',
       'Closing_Date', 'Total_Deposits'],axis=1)
pearson_cor, _ = pearsonr(data_frame['Total_Assets'],data_frame['Estimated_Loss'])
spearman_cor, _ = spearmanr(data_frame['Total_Assets'],data_frame['Estimated_Loss'])

print (f"pearson correlation between total assets and estimated loss is : {pearson_cor:.4f}")
print(f"Spearman correlation between total assets and estimated loss is : {spearman_cor:.4f}")

# pearson correlation between total assets and estimated loss is : 0.4934
# Spearman correlation between total assets and estimated loss is : 0.7554
# we can see there is no close correlation between the total assets and estimated loss
 
 # I'm gonna try a different pair. I want to try to see if there is a correlation between the cities
 #and the closing date 
print ("---------------------")
data_frame = df[['City','Closing_Date']].copy()

#need to enumerate cities 
unique_cities = data_frame['City'].unique()
city_to_num = {city: i for i,city in enumerate(unique_cities)}
data_frame['City'] = data_frame['City'].map(city_to_num)

#need to replace the date closed with a proper datetime data type
#
def convertDateAndTime (myObj):
    myObj=myObj.replace("-","")
    return int(myObj); 
data_frame['Closing_Date'] = data_frame['Closing_Date'].apply(convertDateAndTime)

pearson_cor, _ = pearsonr(data_frame['City'],data_frame['Closing_Date'])
spearman_cor, _ = spearmanr(data_frame['City'],data_frame['Closing_Date'])

print (f"pearson correlation between city and clsoing date is : {pearson_cor:.4f}")
print(f"Spearman correlation between city and closing date is : {spearman_cor:.4f}")

# pearson correlation between city and clsoing date is : -0.6638
# Spearman correlation between city and closing date is : -0.7848
# we can see there is no close correlation between the city and closing date 
#
#
#
# I want to see if there is a seasonality of closing dates

data_frame = df.copy()
data_frame['Closing_Date'] = pd.to_datetime(data_frame['Closing_Date'])
data_frame['Month'] = data_frame['Closing_Date'].dt.month
monthly_closure = data_frame['Month'].value_counts().sort_index()
all_months = pd.Series(range(1, 13))
plt.figure(figsize=(12,6))
monthly_closure.plot(kind='bar',color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Closures')
plt.title('Number of Bank Closures in Each Month')
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(axis='y')
plt.show()

#refer to  PATH = ("figure_2.png") for the plot. 
#
#
#
# I want to see if the distribution of states and the closing dates

unique_states = data_frame['State'].unique()
states_to_num = {state: i for i,state in enumerate(unique_states)}
data_frame['State'] = data_frame['State'].map(states_to_num)

states = data_frame['State'].value_counts().sort_index()
all_states= pd.Series(range(1, 45))
plt.figure(figsize=(12,6))
states.plot(kind='bar',color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Closures')
plt.title('Number of Bank Closures in Each State')
plt.xticks(range(0, 44), ['HI','IL','NH','OH','AR','FL','TX','AZ','MI','CT','GA','TN','LA','CA','WI', 'PA', 'NJ', 'NY' ,'UT' ,'MO' ,'MN' ,'NV' ,'KS', 'WV', 'WA', 'MD' ,'OR' ,'NE', 'CO' ,'NC', 'ID', 'WY' ,'SD' ,'OK' ,'AL' ,'IA' ,'IN' ,'KY' ,'VA' ,'NM', 'SC' ,'MA', 'PR' ,'MS'])
plt.grid(axis='y')
plt.show()

#refer to  PATH = ("figure_3.png") for the plot. 
#
#
#
# I want to see if there are outliers in total deposits and assets

z_score_assets = stats.zscore(data_frame['Total_Assets'])
z_score_deposits = stats.zscore(data_frame['Total_Deposits'])
threshold = 3
outliers_deposits = data_frame[abs(z_score_deposits) > threshold]
outliers_assets = data_frame[abs(z_score_assets) > threshold]
print("---------------------")
print("Outliers in Total_Deposits:",outliers_deposits[['Bank_Name','City',"Total_Deposits"]])
print("\nOutliers in Total_Assets:",outliers_assets[['Bank_Name','City',"Total_Assets"]])

# Three outlier banks:
# 1 - Signature Bank
# 2 - Silicon Valley Bank 
# 3 - Washington Mutual Bank (Including its subsidiary Washington Mutual Bank FSB)
# this also is obvious from the plot
plt.figure(figsize=(8, 6))
plt.scatter(data_frame['Total_Deposits'], data_frame['Total_Assets'])
plt.xlabel('Total_Deposits')
plt.ylabel('Total_Assets')
plt.title('Scatter Plot of Total_Deposits vs. Total_Assets')
plt.show()