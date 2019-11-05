# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:11:22 2019

@author: Adarsh
"""

#logistic regression
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import classification_report

#loading the data
bank=pd.read_csv("E:/ADM/Excelr solutions/DS assignments/logistic regression/bank-full.csv",sep=";")

#In dataset some variables has no importance and unkown data so drop

bank.drop(["education"],inplace=True,axis=1)
bank.drop(["pdays"],inplace=True,axis=1)
bank.drop(["previous"],inplace=True,axis=1)
bank.drop(["poutcome"],inplace=True,axis=1)
bank.drop(["month"],inplace=True,axis=1)
bank.drop(["contact"],inplace=True,axis=1)
bank.drop(["job"],inplace=True,axis=1)

#converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
bank["y"]=lb.fit_transform(bank["y"])

bank=pd.get_dummies(bank)
bank=bank.iloc[:,[5,0,1,2,3,4,6,7,8,9,10,11,12,13,14]]

#EDA
a1=bank.describe()
bank.median()
bank.var()
bank.skew()
plt.hist(bank["age"])
plt.hist(bank["balance"])
plt.hist(bank["duration"])

bank.isna().sum()#No NA values present, so no need to do imputation
bank.isnull().sum()

bank.y.value_counts()#count 0 and 1s
bank.loan_no.value_counts()
bank.loan_yes.value_counts()
bank.housing_yes.value_counts()
cor=bank.corr()

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sb.boxplot(x="y",y="age",data=bank,palette="hls")
sb.boxplot(x="y",y="balance",data=bank,palette="hls")
bank.columns


### Splitting the data into train and test data 
train,test  = train_test_split(bank,test_size = 0.3) # 30% size
train.columns

#model buliding
 
model1=sm.logit('y~age+balance+day+duration+campaign+marital_divorced+marital_married+marital_single+default_no+default_yes+housing_no+housing_yes+loan_no+loan_yes',data=train).fit()
   
   
model1.summary()#Housing variables are insignificant, so remove and build new model
model1.summary2()#AIC:17984.62



#new model without housing
model2=sm.logit('y~age+balance+day+duration+campaign+marital_divorced+marital_married+marital_single+default_no+default_yes+loan_no+loan_yes',data=train).fit()
model2.summary()
model2.summary2()#18701.11 :AIC

#model without default variables
model3=sm.logit('y~age+balance+day+duration+campaign+marital_divorced+marital_married+marital_single+loan_no+loan_yes',data=train).fit()
model3.summary()
model3.summary2()#AIC : 18705.60

#from above observation its seen that by removing insignificant variables AIC  is incresing  and same time more variables becomes insignifacant.
#so better is to perform operation on model1 which has low AIC and we can neglect pvalue as its binary in nature.


#prediction
train_pred = model1.predict(train.iloc[:,1:])
# Creating new column 

# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

#classification report
classification = classification_report(train["train_pred"],train["y"])
'''
              precision    recall  f1-score   support

         0.0       0.98      0.90      0.94     30517
         1.0       0.17      0.56      0.26      1130

    accuracy                           0.89     31647
   macro avg       0.58      0.73      0.60     31647
weighted avg       0.95      0.89      0.92     31647
'''

#confusion matrix
confusion_matrx = pd.crosstab(train.train_pred,train['y'])
confusion_matrx

accuracy_train = (27464+637)/(31647)
print(accuracy_train)#88.79

#ROC CURVE AND AUC
fpr,tpr,threshold = metrics.roc_curve(train["y"], train_pred)

#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

################ AUC #########################

roc_auc = metrics.auc(fpr, tpr)         #0.844 : Excellent model

######################It is a good model with AUC = 0.844 ###############################


#Based on ROC curv we can say that cut-off value = 0.50 is the best value for higher accuracy , by selecting different cut-off values accuracy is decreasing.

# Prediction on Test data set

test_pred = model1.predict(test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test.test_pred,test['y'])

confusion_matrix
accuracy_test = (11741+324)/(13564) 
accuracy_test#88.94


'''
####### Its a Just right model because Test and Train accuracy is same #################

Train accuracy=88.79
Test accuracy=88.94

'''