#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train=features_train[:len(features_train)/1]
labels_train=labels_train[:len(labels_train)/1]




#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score
filt=svm.SVC(kernel='rbf',C=10000)
t0=time()
train=filt.fit(features_train,labels_train)
print train
print("training time: ",round(time()-t0,3),"s")

#t1=time()
#pred=train.predict(features_test)
#print(train.predict(features_test[10]))
#print(train.predict(features_test[26]))
#print(train.predict(features_test[50]))
#print("prediction time: ",round(time()-t1,3),"s")
chris=[]
sara=[]
for i in range(len(features_test)):
  if((train.predict(features_test[i])[0]==1)):
    chris.append(1)
  else:
    sara.append(0)
print len(chris)
print len(sara)
print(accuracy_score(labels_test,pred))
print('a')
#########################################################


