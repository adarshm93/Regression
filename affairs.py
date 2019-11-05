# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:22:32 2019

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
affair=pd.read_csv("E:/ADM/Excelr solutions/DS assignments/logistic regression/affairs.csv")

# Droping first column 
affair.drop(["Unnamed: 0"],inplace=True,axis = 1)
affair=pd.get_dummies(affair)

#converting affair variable into binary
affair.loc[affair.affairs>0,'affairs']=1

#EDA
a1=affair.describe()
affair.median()
affair.var()
affair.skew()
plt.hist(affair["age"])
plt.hist(affair["yearsmarried"])

affair.isna().sum()#No NA values present, so no need to do imputation


affair.affairs.value_counts()#count 0 and 1s
affair.gender_female.value_counts()
affair.gender_male.value_counts()
affair.children_no.value_counts()
cor=affair.corr()

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sb.boxplot(x="affairs",y="age",data=affair,palette="hls")
sb.boxplot(x="affairs",y="yearsmarried",data=affair,palette="hls")

### Splitting the data into train and test data 
aff_train,aff_test  = train_test_split(affair,test_size = 0.3) # 30% size
aff_train.columns


#model buliding
 
model1=sm.logit('affairs~age+yearsmarried+religiousness+education+occupation+rating+gender_female+gender_male+children_no+children_yes',data=aff_train).fit()
model1.summary()#many variables are insignificant, so remove and build new model
model1.summary2()#AIC:436.04

aff_train.drop(["education"],inplace=True,axis=1)
aff_train.drop(["gender_male"],inplace=True,axis=1)
aff_train.drop(["gender_female"],inplace=True,axis=1)
aff_train.drop(["occupation"],inplace=True,axis=1)

#build new model
model2= sm.logit('affairs~age+yearsmarried+religiousness+rating+children_no+children_yes',data=aff_train).fit()
model2.summary()
model2.summary2()#AIC: 433.22

####model2 is better than model1 with low AIC

#prediction
train_pred = model2.predict(aff_train.iloc[:,1:])
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
aff_train["train_pred"] = np.zeros(420)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
aff_train.loc[train_pred>0.5,"train_pred"] = 1

#classification report
classification = classification_report(aff_train["train_pred"],aff_train["affairs"])
'''
              precision    recall  f1-score   support

         0.0       0.96      0.77      0.86       398
         1.0       0.10      0.45      0.16        22

    accuracy                           0.75       420
   macro avg       0.53      0.61      0.51       420
weighted avg       0.92      0.75      0.82       420
'''
#confusion matrix
confusion_matrx = pd.crosstab(aff_train.train_pred,aff_train['affairs'])
confusion_matrx

accuracy_train = (307+10)/(420)
print(accuracy_train)#75.47

#ROC CURVE AND AUC
fpr,tpr,threshold = metrics.roc_curve(aff_train["affairs"], train_pred)

#PLOT OF ROC
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

################ AUC #########################

roc_auc = metrics.auc(fpr, tpr)         #0.705 : Good model

######################It is a good model with AUC = 0.705 ###############################


#Based on ROC curv we can say that cut-off value = 0.50 is the best value for higher accuracy , by selecting different cut-off values accuracy is decreasing.

# Prediction on Test data set

test_pred = model2.predict(aff_test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
aff_test["test_pred"] = np.zeros(181)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
aff_test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(aff_test.test_pred,aff_test['affairs'])

confusion_matrix
accuracy_test = (128+7)/(181) 
accuracy_test#74.58


'''
####### Its a Just right model because Test and Train accuracy is nearly same #################

Train accuracy=75.47
Test accuracy=74.58

'''