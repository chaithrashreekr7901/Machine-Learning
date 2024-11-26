 import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#print(sns.get_dataset_names())

data=sns.load_dataset("iris")
print(data.head(2))
print(data.tail(2))
# print(data.info())
print(data.isnull().sum())
print(data['species'].value_counts())

#features   x
x=np.array(data.iloc[:,0:4])
#print(x)
print(x.shape)

#target y
y=np.array(data.iloc[:,4])
#print(y)
print(y.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.80,random_state=2)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=4,p=2)  #p is distance measures(eculidian=p2 and mancadian=p1)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
# print(ypred) #predicted
# print(ytest)  #actual
count=0
for i in range(len(ytest)):
    if ypred[i]==ytest[i]:
        count=count+1
# print(count)
# print(len(ytest))
print(count/len(ytest))

from sklearn.metrics import accuracy_score
a=accuracy_score(ytest,ypred)
print(a)

import joblib
joblib.dump(model,"iris.pkl")

mymod=joblib.load("C:/Users/MCA/PycharmProjects/pythonProject/Harshitha/iris.pkl")
print(data.head())
print(mymod.predict([[4.6,3.2,1.3,0.2]]))   