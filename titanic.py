import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#print(sns.get_dataset_names())

data=sns.load_dataset("titanic")
#print(data.head())  #print first 5 rows of dataset
#print(len(data))  #len is 891

#prints 0 as dead and 1 as survived
#print(data.info())
#print(data['survived'].value_counts())

#sns.countplot(x=data['survived']) #(output a)
# sns.countplot(x=data['survived'],hue=data['deck']) #(output b)
# plt.show() #bar graph of survived ppl (output a) and all the remaining instances(output b)

#print(data.isnull().sum()) #print the null values

#survived is target value and rest is considered as features(always take target value of integer type)
print(data.columns)
cols=['fare','class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone']
data_new=data.drop(cols,axis=1) #drop the above columns listed(1=column,0=rows)

print(data_new.head)
print(data_new.isnull().sum()) #age = 177 (so preprocess the data)

mean_age=data_new['age'].mean()
#print(mean_age) #0/p 29.69911764705882
mean_age=np.round(mean_age,2)
print(mean_age) #o/p 29.7

data_new['age']=data_new['age'].fillna(mean_age)
print(data_new.isnull().sum())

data_new=data_new.dropna()
print(data_new.isnull().sum()) #all data preprocessed
# print(len(data_new)) # len after prepro 889
#print(data_new.head())

#converting string value as numeric
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data_new['sex']=enc.fit_transform(data_new['sex'])
#print(data_new.head())
data_new['embarked']=enc.fit_transform(data_new['embarked'])
#print(data_new.head())

#features x
x=np.array(data_new.iloc[:,1:])
#print(x.shape)  #already in 2D
#target y
y=np.array(data_new.iloc[:,0])
#print(y.shape)  #its in 1D (target can be 1D or 2D both)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=3)
#print(pd.DataFrame(y).value_counts())
#print(pd.DataFrame(ytrain).value_counts())

#applying the ml model/alg
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3,p=2)  #n_neighbors=3 is value of k and p is distance measures(eculidian=p2 and mancadian=p1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
print(ypred) #predicted o/p
print(ytest) #actual o/p
print(ytest[1]==ypred[1])
count=0
for i in range(len(ytest)):
       if ypred[i]==ytest[i]:
              count=count+1
# print(count) #print the no of correct predicted value ie 137
# print(len(ytest)) #print the total no of sample in ytest 178
print(count/len(ytest)*100) #method1 :- to find accuracy

from sklearn.metrics import accuracy_score
a=accuracy_score(ytest,ypred)
#print(a) #method 2:- to find accuracy can use any one of among

import joblib
joblib.dump(model,"titanic1.pkl")
mymodel=joblib.load("C:/Users/Admin/PycharmProjects/pythonProject/iot_progs/titanic1.pkl")
print(data_new.head())
print(mymodel.predict([[3,1,22,1,0,2]])) #give any new sample to predict the value