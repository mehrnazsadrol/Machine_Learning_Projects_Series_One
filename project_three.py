import pandas as pd
import matplotlib.pyplot as pyplt
from scipy import stats
import numpy as np

df = pd.read_csv("healthcare-dataset-stroke-data.csv")


#let's get some information about the data 
pd.set_option('display.max_columns', None)
print (df.describe()); 
print (df.info)
print (df.dtypes)


#clean the data 
df.dropna(inplace = True)
print (df.isnull().sum())

# I want to check the outliers 
selected_cols = ['bmi','avg_glucose_level','age']
colors = ['tab:red','tab:green','tab:pink','tab:blue']
pyplt.figure(figsize=(12,6))
for i, col in enumerate(selected_cols, 1):
    pyplt.subplot(1, 3, i)
    pyplt.boxplot(df[col],vert=True,patch_artist=True)
    pyplt.title(f'Box plot of {col}')
    pyplt.ylabel('Values')

pyplt.show();
#refer to  PATH = ("figure_1_3.png") for the plot.

#now it's obvious that we need to get rid of outliers. Now the shape of the data
#before removing the outliers is as folows 
pyplt.figure(figsize=(16,6))
for i, col in enumerate(selected_cols,1):
    pyplt.subplot(1,3,i)
    pyplt.hist(df[col], bins=100, color = colors[i])
    pyplt.ylabel('Value')
    pyplt.title(f'Distribution of {col} before removing the outliers')
pyplt.tight_layout()
pyplt.show();

#refer to  PATH = ("figure_2_3.png") for the plot.

#removing the outliers using the z method
df_no_outliers = df.copy();
for col in selected_cols:
   z_score = np.abs(stats.zscore(df[col]))
   df_no_outliers[col] = np.where(z_score < 3, df_no_outliers[col] , np.nan)

#now it's obvious that we need to get rid of outliers. Now the shape of the data
#before removing the outliers is as folows 
pyplt.figure(figsize=(16,6))
for i, col in enumerate(selected_cols,1):
    pyplt.subplot(1,3,i)
    pyplt.hist(df_no_outliers[col], bins=100, color = colors[i])
    pyplt.ylabel('Value')
    pyplt.title(f'Distribution of {col} after removing the outliers')
pyplt.tight_layout();
pyplt.show();

#refer to  PATH = ("figure_3_3.png") for the plot.
#
#
#
#some visualizaitons here under to better understand the data 

#pie chart to show the female to make ratio
female_count = df[ df['gender'] == 'Female'].shape[0]
print(female_count)
male_count = df[df['gender'] == 'Male'].shape[0]
print (male_count)
total_count = female_count + male_count

labels = ['Female','Male']
size = [(female_count/total_count), (male_count/total_count)]
colours = ['lightgreen','darkgreen']
pyplt.pie (size,labels=labels,autopct='%1.1f%%',colors = colours,startangle=145)
pyplt.title('Proportion of Females in the dataset')
pyplt.show();
#refer to  PATH = ("figure_4_3.png") for the plot.

