import pandas as pd

def extract_title(df):
    df['Title']=df.Name.apply(lambda x: x.split(' ')[1])
    return(df)
    
def extract_block_letter(df):
    df.Cabin=df.Cabin.fillna('U')
    df.Cabin=df.Cabin.astype(str)
    df['Block']=df.Cabin.apply(lambda c:c[0])
    return(df)
            
def fill_missing_ages(df):
    titles=['Mr', 'Mrs', 'Miss', 'Master']
    for i in range(0, df.shape[0]):
        if pd.isnull(df.Age.iloc[i])==True and df.Title.iloc[i] in titles:
            df['Age'].iloc[i]=int(df['Age'].loc[df['Title']==df.Title.iloc[i]].mean())
        elif pd.isnull(df.Age.iloc[i])==True and df.Title.iloc[i] not in titles:
            df['Age'].iloc[i]=int(df['Age'].loc[df['Sex']==df.Sex.iloc[i]].mean())
    return(df)

def create_age_buckets(df):
    bins = (0, 6, 13, 18, 25, 35, 60, 100)
    group_names = ['Infant', 'Child', 'Teen', 'Student', 'Young_Adult', 'Adult', 'Elderly']
    buckets = pd.cut(df.Age, bins, labels=group_names)
    df.Age = buckets
    return(df)

def transform_features(df):
    df=extract_title(df)
    df=extract_block_letter(df)
    df=fill_missing_ages(df)
    df=create_age_buckets(df)
    return(df)
    
def create_x_set(df):
    df.drop(['Name','Ticket','Fare','Cabin','Embarked','Title'], axis=1, inplace=True)
    X_df=df[['Pclass','Age','Sex','SibSp','Parch', 'Block']]
    X_df=pd.concat([X_df.drop(['Pclass','Sex','Block', 'Age'], axis=1), pd.get_dummies(X_df[['Pclass','Sex', 'Block','Age']], drop_first=True)],axis=1)
    return(X_df)
