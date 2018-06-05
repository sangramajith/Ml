import pandas as pd
import matplotlib.pyplot as plt

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

combine=[train_df,test_df]

train_df.describe(include=['O'])
train_df.describe()


train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived'
,ascending=False)

train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean()

train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()

train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()

mean1=train_df.Age.dropna().mean()
mean2=test_df.Age.dropna().mean()
train_df['Age']=train_df['Age'].fillna(mean1)
test_df['Age']=test_df['Age'].fillna(mean2)

freq1=train_df.Embarked.dropna().mode()
train_df['Embarked']=train_df['Embarked'].fillna(freq1)
test_df['Embarked']=test_df['Embarked'].fillna(freq1)
    

for dataset in combine:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    
X_train=train_df.drop(['PassengerId','Survived','Name','Parch','SibSp','Ticket','Fare','Cabin'],axis=1)
X_test=test_df.drop(['PassengerId','Name','Parch','SibSp','Ticket','Fare','Cabin'],axis=1)

Y_train=train_df["Survived"]

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log



    
