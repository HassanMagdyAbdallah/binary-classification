# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:48:54 2019

@author: Hassan
"""
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import copy

def Floating(obj):
    return obj.convert_objects(convert_numeric=True)

############################## Data Preprocessing ############################
    
#Load Training and Validation Sets
train_data = pd.read_csv("training.csv",sep = ";" )
validation_data = pd.read_csv("validation.csv",sep = ";" )
print (train_data.dtypes)
train_objs_num = len(train_data)
dataset = pd.concat(objs=[train_data, validation_data], axis=0)

# #Convert from Object type to Float type
dataset['variable2'] = Floating(dataset['variable2'])
dataset['variable3'] = Floating(dataset['variable3'])
dataset['variable8'] = Floating(dataset['variable8'])
dataset['variable11'] = Floating(dataset['variable11'])
dataset['variable15'] = Floating(dataset['variable15'])
dataset['variable19'] = Floating(dataset['variable19'])
dataset = pd.get_dummies(dataset[['variable1','variable4','variable5','variable6','variable7',
                            'variable9','variable10','variable12','variable13','variable18']])

train_data = copy.copy(dataset[:train_objs_num])
validation_data = copy.copy(dataset[train_objs_num:])


#print some information
train_data.head(5)
print(train_data.shape)
print (train_data.dtypes)

#plotting
train_data.plot()
plt.show()

#Extract Features and Labels
features = train_data.iloc[:,:-1].values
Label = train_data.iloc[:,41].values
V_features = validation_data.iloc[:,:-1].values
V_Label = validation_data.iloc[:,41].values

#print some information of features
print(features.shape)
print(Label.shape)
print(features[1,:])
print(Label[40])

#average = np.mean(features,0)
#print(average)

#feature Scaling
sc = StandardScaler()
features = sc.fit_transform(features)
V_features = sc.fit_transform(V_features)

#fit the model
classifier = LogisticRegression()
classifier = classifier.fit(features,Label) 
y_pred = classifier.predict(V_features)

#Confusion Matrix
cm = confusion_matrix(V_Label,y_pred)
#accuracy 
accuracy = classifier.score(V_features,V_Label)
#Area Under ROC Curve
auc = cross_val_score(classifier, features, Label, scoring='roc_auc').mean()
#Classificarion Report
report = classification_report(V_Label, y_pred)

# fit another model
from sklearn.linear_model import SGDClassifier
classifier2 = SGDClassifier(loss="hinge", penalty="l2",n_iter=10000)
classifier2 = classifier2.fit(features,Label)
y_pred2 = classifier2.predict(V_features)

#Confusion Matrix
cm2 = confusion_matrix(V_Label,y_pred2)
#accuracy 
accuracy2 = classifier2.score(V_features,V_Label)
#Area Under ROC Curve
auc2 = cross_val_score(classifier2, features, Label, scoring='roc_auc').mean()
#Classificarion Report
report2 = classification_report(V_Label, y_pred2)