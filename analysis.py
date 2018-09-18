import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

train_df=pd.read_csv('titanic_data/train.csv', encoding='latin-1')

#checking for columns with null values
train_df.isnull().sum()

#checking that the training set is well balanced
train_df.groupby('Survived').count().PassengerId

#creating Title and Block feature based on preexisting Name and Cabin columns
train_df['Title']=train_df['Name']
for i in range(0,train_df.shape[0]):
    name=train_df['Name'].iloc[i]
    train_df['Title'].iloc[i]=name[name.find(",")+2:name.find(".")]

train_df['Block']= "U"
for i in range(0, train_df.shape[0]):
    if pd.isnull(train_df.Cabin.iloc[i])==False:
        train_df['Block'].iloc[i]=train_df['Cabin'].iloc[i][0]
        
#Passenger class is a good predictor of the outcome
Pclass_plot=sns.barplot(x='Pclass',y='Survived',data=train_df)
plt.savefig('analysis_plots/Pclass_vs_Survival.pdf')
plt.clf()

#Gender of the passenger is a good predictor of survival
gender_plot=sns.barplot(x='Sex', y='Survived', data=train_df)
plt.savefig('analysis_plots/Sex_vs_Survival.pdf')
plt.clf()

#filling missing ages
titles=['Mr', 'Mrs', 'Miss', 'Master']
for i in range(0, train_df.shape[0]):
    if pd.isnull(train_df.Age.iloc[i])==True and train_df.Title.iloc[i] in titles:
        train_df['Age'].iloc[i]=int(train_df['Age'].loc[train_df['Title']==train_df.Title.iloc[i]].mean())
    elif pd.isnull(train_df.Age.iloc[i])==True and train_df.Title.iloc[i] not in titles:
        train_df['Age'].iloc[i]=int(train_df['Age'].loc[train_df['Sex']==train_df.Sex.iloc[i]].mean())
    train_df['Age'].iloc[i]=math.ceil(train_df['Age'].iloc[i])

#age seems to be an okay predictor of survival
age_plot=sns.barplot(x='Age', y='Survived', data=train_df, ci=None)
plt.savefig('analysis_plots/Age_vs_Survival.pdf')
plt.clf()

#the number of siblings and spouse onboard with each passenger seems to be a good indicator of survival
sibsp_plot=sns.barplot(x='SibSp', y='Survived', data=train_df, ci=None)
plt.savefig('analysis_plots/SibSp_vs_Survival.pdf')
plt.clf()

#the number of parents and children onboard with each passenger seems to be a good indicator of survival
parch_plot=sns.barplot(x='Parch', y='Survived', data=train_df, ci=None)
plt.savefig('analysis_plots/Parch_vs_Survival.pdf')
plt.clf()

#The deck letter (A,B,C...) in cabin numbers seems to be a good indicator of survival
block_plot=sns.barplot(x='Block', y='Survived', data=train_df, ci=None)
plt.savefig('analysis_plots/block_vs_Survival.pdf')
plt.clf()
