import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from preprocess import *

train_df=pd.read_csv('titanic_data/train.csv', encoding='latin-1')
test_df=pd.read_csv('titanic_data/test.csv', encoding='latin-1')

#pre-processing training set and test set
train_df=transform_features(train_df)
test_df=transform_features(test_df)

X_train_df=create_x_set(train_df)
Y_train_df=train_df['Survived']

#splitting training set in two to evaluate accuracy of the logistic regression model
LogReg=LogisticRegression()
x_train,x_test, y_train, y_test=train_test_split(X_train_df, Y_train_df, test_size=0.3)
LogReg.fit(x_train,y_train)
LogReg.score(x_test,y_test)

#training logistic regression model
LogReg.fit(X_train_df, Y_train_df)

#validating with k-fold cross validation to make sure the model generalises well
scores = cross_val_score(LogReg, X_train_df, Y_train_df, cv=10)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std()*2))

#preprocessing test set
X_df=create_x_set(test_df)
X_df['Block_T']=0
X_df=X_df[list(X_train_df.columns.values)]

#Predicting survival for test set
Y_pred=LogReg.predict(X_df)
output=pd.DataFrame(Y_pred, index=test_df.PassengerId, columns=['Survived'])
output.to_csv('LogisticRegression_Output.csv')
