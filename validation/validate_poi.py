#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
filt=DecisionTreeClassifier()
#training on whole data set and predicting from that only which is not very accurate.
filt=filt.fit(features,labels)
pred=filt.predict(features)
from sklearn.metrics import accuracy_score
print('Accuracy',accuracy_score(labels,pred))
# accuracy: 0.989473

##Lets make the split in the data with test_train_split
from sklearn.model_selection import train_test_split
#We are taking 30% data for test and putting random_state value 43.
#Take care in order of features and labels
labels_train,labels_test,features_train,features_test=train_test_split(labels,features,test_size=0.3,random_state=42)
filt_n=DecisionTreeClassifier()
filt_n=filt_n.fit(features_train,labels_train)
pred_n=filt_n.predict(features_test)
print('updated Accuracy',accuracy_score(labels_test,pred_n))
# updated Accuracy: 0.724137
