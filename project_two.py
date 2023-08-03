import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

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


#since I can't find a correlation between the columns. I will try to do time series analysis

data_frame = df.copy()
data_frame['Closing_Date'] = pd.to_datetime(data_frame['Closing_Date'])
data_frame.sort_values(by='Closing_Date',inplace=True)

plt.figure(figsize=(12,6))
plt.plot(data_frame['Closing_Date'],data_frame['Total_Deposits'],label='Total Deposits')
plt.plot(data_frame['Closing_Date'],data_frame['Total_Assets'],label='Total Assets')
plt.xlabel('Closing date')
plt.ylabel('Amount')
plt.title('Total deposits and total assets over time')
plt.legend()
plt.show()

#refer to  PATH = ("figure_one.png") for the plot. 