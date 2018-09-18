import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

train_df=pd.read_csv('titanic_data/train.csv', encoding='latin-1')
test_df=pd.read_csv('titanic_data/test.csv', encoding='latin-1')

#pre-processing training set
train_df['Title']=train_df['Name']
for i in range(0,train_df.shape[0]):
    name=train_df['Name'].iloc[i]
    train_df['Title'].iloc[i]=name[name.find(",")+2:name.find(".")]
    
#filling missing ages
titles=['Mr', 'Mrs', 'Miss', 'Master']
for i in range(0, train_df.shape[0]):
    if pd.isnull(train_df.Age.iloc[i])==True and train_df.Title.iloc[i] in titles:
        train_df['Age'].iloc[i]=int(train_df['Age'].loc[train_df['Title']==train_df.Title.iloc[i]].mean())
    elif pd.isnull(train_df.Age.iloc[i])==True and train_df.Title.iloc[i] not in titles:
        train_df['Age'].iloc[i]=int(train_df['Age'].loc[train_df['Sex']==train_df.Sex.iloc[i]].mean())

train_df['Block']= 'U'
for i in range(0, train_df.shape[0]):
    if pd.isnull(train_df.Cabin.iloc[i])==False:
        train_df['Block'].iloc[i]=train_df['Cabin'].iloc[i][0]

train_df.drop(['Name','Ticket','Fare','Cabin','Embarked','Title'], axis=1, inplace=True)
X_train_df=train_df[['Pclass','Age','Sex','SibSp','Parch', 'Block']]
X_train_df=pd.concat([X_train_df.drop(['Pclass','Sex','Block'], axis=1), pd.get_dummies(X_train_df[['Pclass','Sex', 'Block']].astype(str), drop_first=True)],axis=1)
Y_train_df=train_df['Survived']

#splitting training set in two to evaluate accuracy of the logistic regression model
LogReg=LogisticRegression()
x_train,x_test, y_train, y_test=train_test_split(X_train_df, Y_train_df, test_size=0.3)
LogReg.fit(x_train,y_train)
LogReg.score(x_test,y_test)

#training logistic regression model
LogReg.fit(X_train_df, Y_train_df)

#pre-processing test set
test_df['Title']=test_df['Name']
for i in range(0,test_df.shape[0]):
    name=test_df['Name'].iloc[i]
    test_df['Title'].iloc[i]=name[name.find(",")+2:name.find(".")]
    
test_df['Block']= 'U'
for i in range(0, test_df.shape[0]):
    if pd.isnull(test_df.Cabin.iloc[i])==False:
        test_df['Block'].iloc[i]=test_df['Cabin'].iloc[i][0]
for i in range(0, test_df.shape[0]):
    if pd.isnull(test_df.Age.iloc[i])==True and test_df.Title.iloc[i] in titles:
        test_df['Age'].iloc[i]=int(test_df['Age'].loc[test_df['Title']==test_df.Title.iloc[i]].mean())
    elif pd.isnull(test_df.Age.iloc[i])==True and test_df.Title.iloc[i] not in titles:
        test_df['Age'].iloc[i]=int(test_df['Age'].loc[test_df['Sex']==test_df.Sex.iloc[i]].mean())

test_df.drop(['Name','Ticket','Fare','Cabin','Embarked','Title'], axis=1, inplace=True)
X_test_df=test_df[['Pclass','Age','Sex','SibSp','Parch', 'Block']]
X_test_df=pd.concat([X_test_df.drop(['Pclass','Sex','Block'], axis=1), pd.get_dummies(X_test_df[['Pclass','Sex', 'Block']].astype(str), drop_first=True)],axis=1)
X_test_df['Block_T']=0
X_test_df=X_test_df[list(X_train_df.columns.values)]

#Predicting survival for test set
Y_pred=LogReg.predict(X_test_df)
output=pd.DataFrame(Y_pred, index=test_df.PassengerId, columns=['Survived'])
output.to_csv('LogisticRegression_Output.csv')
