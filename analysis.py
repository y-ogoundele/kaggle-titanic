import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import *

train_df=pd.read_csv('titanic_data/train.csv', encoding='latin-1')

#checking for columns with null values
train_df.isnull().sum()

#checking that the training set is well balanced
train_df.groupby('Survived').count().PassengerId

train_dfdf=extract_title(train_df)
train_df=extract_block_letter(train_df)
train_df=fill_missing_ages(train_df)
        
#Passenger class is a good predictor of the outcome
Pclass_plot=sns.barplot(x='Pclass',y='Survived',data=train_df)
plt.savefig('analysis_plots/Pclass_vs_Survival.pdf')
plt.clf()

#Gender of the passenger is a good predictor of survival
gender_plot=sns.barplot(x='Sex', y='Survived', data=train_df)
plt.savefig('analysis_plots/Sex_vs_Survival.pdf')
plt.clf()

#age doesn't seem to be a great predictor of survival
age_plot=sns.barplot(x='Age', y='Survived', data=train_df, ci=None)
plt.savefig('analysis_plots/Age_vs_Survival.pdf')
plt.clf()

#I decided to group ages into larger buckets
train_df=create_age_buckets(train_df)
age_bucket_plot=sns.barplot(x='Age', y='Survived', hue='Sex', data=train_df, ci=None)
plt.savefig('analysis_plots/Age_bucket_vs_Survival.pdf')
plt.clf()


#the number of siblings and spouse onboard with each passenger seems to be a good indicator of survival
sibsp_plot=sns.barplot(x='SibSp', y='Survived', hue='Sex', data=train_df, ci=None)
plt.savefig('analysis_plots/SibSp_vs_Survival.pdf')
plt.clf()

#the number of parents and children onboard with each passenger seems to be a good indicator of survival
parch_plot=sns.barplot(x='Parch', y='Survived', hue='Sex', data=train_df, ci=None)
plt.savefig('analysis_plots/Parch_vs_Survival.pdf')
plt.clf()

#The deck letter (A,B,C...) in cabin numbers seems to be a good indicator of survival
block_plot=sns.barplot(x='Block', y='Survived', hue='Sex', data=train_df, ci=None)
plt.savefig('analysis_plots/block_vs_Survival.pdf')
plt.clf()
