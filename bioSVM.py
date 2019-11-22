#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:26:10 2019

@author: nageshsinghchauhan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Load dataset
Train_Data = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/bioinformatics/data_set_ALL_AML_train.csv")
Test_Data = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/bioinformatics/data_set_ALL_AML_independent.csv")
labels = pd.read_csv("/Users/nageshsinghchauhan/Downloads/ML/bioinformatics/actual.csv", index_col = 'patient')
#check for nulls
print(Train_Data.isna().sum().max())
print(Test_Data.isna().sum().max())

#drop 'call' and  columns
cols = [col for col in Test_Data.columns if 'call' in col]
test = Test_Data.drop(cols, 1)
cols = [col for col in Train_Data.columns if 'call' in col]
train = Train_Data.drop(cols, 1)
#Join all the data
patients = [str(i) for i in range(1, 73, 1)]
df_all = pd.concat([train, test], axis = 1)[patients]
#transpose rows and columns
df_all = df_all.T

df_all["patient"] = pd.to_numeric(patients)
labels["cancer"]= pd.get_dummies(labels.cancer, drop_first=True)
# add the cancer column to train data

Data = pd.merge(df_all, labels, on="patient")

#Train_Data, Test_Data = Data.iloc[:39,:], Data.iloc[39:,:]

#X -> matrix of independent variable
#y -> vector of dependent variable
X, y = Data.drop(columns=["cancer"]), Data["cancer"]

#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# perform PCA on the Data to reduce the amount of features
from sklearn.decomposition import PCA
pca = PCA(n_components = 38) # use fist ~20 components to make svc
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

"""
#how to choose the value of K
total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1
#k = 38
"""
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100
plt.bar(range(38), cum_sum)
plt.ylabel("Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.title("Around 90% of variance is explained by the First 22 columns ")


# do a grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

search = GridSearchCV(SVC(), parameters, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

best_accuracy = search.best_score_ #to get best score
best_parameters = search.best_params_ #to get best parameters
# select best svc
best_svc = search.best_estimator_

#build SVM model with best parameters
model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score:', accuracy_score(y_test, y_pred))
#confusion matrix
cm = confusion_matrix(y_test, y_pred)

