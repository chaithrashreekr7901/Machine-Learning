from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#load dataset
cancer_data=datasets.load_breast_cancer()
print(cancer_data)
print(cancer_data['target'])

#split the data
xtrain,xtest,ytrain,ytest=train_test_split(cancer_data.data,cancer_data.target,test_size=0.4,random_state=209)
#generate the model
cls=svm.SVC(kernel="linear")
#train the model
cls.fit(xtrain,ytrain)
pred=cls.predict(xtest)
print("Accuracy:",metrics.accuracy_score(ytest,y_pred=pred))
print("Precision:",metrics.precision_score(ytest,y_pred=pred))
print("Recall:",metrics.recall_score(ytest,y_pred=pred))
print((metrics.classification_report(ytest,y_pred=pred)))