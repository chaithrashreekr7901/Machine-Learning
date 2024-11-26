import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/movies.csv")

#print(type(data))
#print(data)
#print(data.head(3))
#print(data.tail())
#print(data.sample(random_state=3))  #or print(data.sample(3)) for any three random rows

#print(data.info()) #get info.about all the dataset
#print(data.isnull().sum()) #get info. about all the null value or not clean values

#working with missing values
# data_1=data.dropna(axis=0,how="all")
# print(len(data))
# print(len(data_1))

# data_1=data.dropna(axis=0,how="any")  #if any value is null then it deletes the entire row
# print(len(data_1))

#data_1=data.dropna(axis=0,how="all",subset=["GENRE"])
# print(len(data_1))

#print(data.isnull().sum())  #there are 9539 null values in gross column,so perform delete
# data=data.drop(["Gross"],axis=1)  #deleted column ie Gross.for column(axis=1) and for rows(axis=0)
# print(data.isnull().sum())

print(data['VOTES'])
data['VOTES']=data['VOTES'].fillna("0")
print(data['VOTES'])
print(data.isnull().sum())

#print(data['RunTime'])
meanrt=data['RunTime'].mean()
#print(meanrt)
meanrt=round(meanrt,1)
#print(meanrt)
data['RunTime']=data['RunTime'].fillna(meanrt)
#print(data['RunTime'])
print(data.isnull().sum())

#print(data['RATING'])
meanrtng=data['RATING'].mean()
#print(meanrtng)
meanrtng=round(meanrtng,1)
#print(meanrtng)
data['RATING']=data['RATING'].fillna(meanrtng)
#print(data['RATING'])
print(data.isnull().sum())

#print(data['GENRE'])
data['GENRE']=data['GENRE'].fillna("Comedy")
#print(data['GENRE'])
print(data.isnull().sum())

#print(data['YEAR'])
data['YEAR']=data['YEAR'].fillna(1999)
#print(data['YEAR'])

print(data.isnull().sum())