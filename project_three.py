
## Analyzing stroke prediction data set. Different visualization methods used to get insights 
## to the dataset and logisticregression model used to predict strokes on patients 
## Author : Mehrnaz Sadroleslami 
## Date : Aug 4th, 2023
## DataSet : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

import pandas as pd
import matplotlib.pyplot as pyplt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("healthcare-dataset-stroke-data.csv")


#let's get some information about the data 
pd.set_option('display.max_columns', None)
# print (df.describe()); 
# print (df.info)
# print (df.dtypes)


#clean the data 
df.dropna(inplace = True)
# print (df.isnull().sum())

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


fig, axes = pyplt.subplots(nrows=2, ncols=4, figsize=(15, 10))
axes = axes.flatten()
#pie chart to show the female to make ratio

female_count = df[ df['gender'] == 'Female'].shape[0]
male_count = df[df['gender'] == 'Male'].shape[0]
total_count = female_count + male_count
labels = ['Female','Male']
size = [(female_count/total_count), (male_count/total_count)]
colours = ['pink','lightgreen']
axes[0].pie (size,labels=labels,autopct='%1.1f%%',colors = colours,startangle=145)
axes[0].set_title('Gender of patients')



total = df['work_type'].value_counts()
size = total.values
labels = total.index
colors = ['lightcoral', 'lightblue', 'lightskyblue', 'lightgreen', 'pink']
axes[1].pie(size, labels=labels, colors=colors, autopct='%1.1f%%',  startangle=145)
axes[1].set_title('Work Type of Patients')


smoking_status = df['smoking_status'].value_counts()
labels = smoking_status.index
sizes = smoking_status.values
colors = ['lightcoral', 'lightblue','lightgreen']
axes[2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=145)
axes[2].set_title('Smoking Status of Patients')


total = df['heart_disease'].value_counts()
size = total.values
labels = ['No','Yes']
colors = ['lightcoral', 'lightblue']
axes[3].pie(size, labels=labels, colors=colors, autopct='%1.1f%%',  startangle=145)
axes[3].set_title('Heart disease history')




total = df['work_type'].value_counts()
size = total.values
labels = total.index
colors = ['lightcoral', 'lightblue', 'lightskyblue', 'lightgreen', 'pink']
axes[4].pie(size, labels=labels, colors=colors, autopct='%1.1f%%',  startangle=145)
axes[4].set_title('Work Type of Patients')


bmi_categories = ['Underweight', 'Normal Weight', 'Overweight', 'Obese']
bmi_thresholds = [18.5, 24.9, 29.9]
df['bmi_category'] = pd.cut(df['bmi'], bins=[0] + bmi_thresholds + [float('inf')], labels=bmi_categories)
total = df['bmi_category'].value_counts()
size = total.values
labels = total.index
colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'pink']
axes[5].pie(size, labels=labels, colors=colors, autopct='%1.1f%%',  startangle=145)
axes[5].set_title('BMI categories')


married_counts = df['ever_married'].value_counts()
labels = married_counts.index
sizes = married_counts.values
colors = ['lightcoral', 'lightblue']
axes[6].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=145)
axes[6].set_title('Marritual Status of Patients')

fig.delaxes(axes[7])
fig.suptitle('Stroke Data Visualizations', fontsize=10)
pyplt.tight_layout();
pyplt.show()

#refer to  PATH = ("figure_4_3.png") for the plot.



# here starts the machine learning part of this project !!
#let the fun start
print (df.columns)
label_encoder = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop(['stroke','bmi_category'], axis=1)
y =df['stroke']

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=42)

model = LogisticRegression(random_state= 42)
model.fit(train_X,train_y)
y_prediction = model.predict(val_X)

accuracy = accuracy_score(val_y, y_prediction)
print("Accuracy of prediction with the logistic regression model is :", accuracy)
class_report = classification_report(val_y, y_prediction)
print("Classification Report:")
print (class_report)


#
#result: 
#Accuracy of prediction with the logistic regression model is : 0.9495114006514658
#
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.95      1.00      0.97      1165
#            1       1.00      0.02      0.03        63

#     accuracy                           0.95      1228
#    macro avg       0.97      0.51      0.50      1228
# weighted avg       0.95      0.95      0.93      1228
#
#
#
#
# you can see that the model is predicting the strokes on the patients based the on the
#given data with a high accuracy. 

