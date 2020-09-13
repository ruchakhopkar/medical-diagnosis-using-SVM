# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 01:15:17 2020

@author: rucha
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

column_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df=pd.read_csv('diabetes.csv', names=column_names)
print(df.shape)

X=df.iloc[:,:8]
y=df['Outcome']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

clf=svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_train)
print(accuracy_score(y_train, y_pred))

for k in ('linear', 'poly', 'rbf', 'sigmoid'): #Hyperparameter Optimization
    clf=svm.SVC(kernel=k)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_train)
    print(k)
    print(accuracy_score(y_train, y_pred))
    
clf=svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
patient=np.array([[1., 200., 75., 40., 0., 45., 1.5, 20.]])
patient=scaler.transform(patient)
print(clf.predict(patient))


X_test=scaler.transform(X_test)
y_pred=clf.predict(X_test)
print(accuracy_score( y_test, y_pred))

y_zeros=np.zeros(y_test.shape)
print(accuracy_score( y_test, y_zeros)) #unbalanced data

print(classification_report(y_test, y_pred))

#if precision close to 1- we are trying to avoid false positives
#if recall close to 1- we are trying to avoid false negatives
# F1 score combines the 2 but focuses more on the one that is worse
    