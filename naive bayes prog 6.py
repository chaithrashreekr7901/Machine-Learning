  import pandas as pd
import numpy as py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df=pd.read_csv("C:/Users/Admin/Downloads/ml dataset/placement.csv")
#print(df)
#print(df.info())
print(df.isnull().sum())
#print(df.shape)
sns.countplot(x="status",hue="gender",data=df)
plt.show()

df.drop(['sl_no','gender','ssc_b','hsc_b','hsc_s','degree_t','specialisation'],axis=1,inplace=True)
#print(df.head(2))

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
df['workex']=enc.fit_transform(df['workex'])
#print(df)

#x features
x=df.drop('status',axis=1)
print(x.shape)
#y target
y=df['status']
print(y.shape)

#train and test the model
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)

#apply naive bayes
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)

#prediction
ypred=model.predict(xtest)
df1=pd.DataFrame({'Actual Status':ytest,'Predicted Status':ypred})
#print(df1)

print(classification_report(ytest,ypred))
print(confusion_matrix(ytest,ypred))
print(accuracy_score(ytest,ypred))

import joblib
joblib.dump(model,"plment.pkl")

mymod=joblib.load("C:/Users/Admin/PycharmProjects/pythonProject/iot_progs/plment.pkl")
print(df.head())
print(mymod.predict([[67.00,83.00,57.00,1,56.00,83.00]]))