#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

## code from the last problem validation\validation_poi
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#We are taking 30% data for test and putting random_state value 43.
#Take care in order of features and labels
labels_train,labels_test,features_train,features_test=train_test_split(labels,features,test_size=0.3,random_state=42)
filt_n=DecisionTreeClassifier()
filt_n=filt_n.fit(features_train,labels_train)
pred_n=filt_n.predict(features_test)
print('updated Accuracy',accuracy_score(labels_test,pred_n))

# For total no. of POI in the test set we can find 1 in the prediction metrix pred_n
temp=[]
for i in range(len(pred_n)):
    if pred_n[i]==1:
        temp.append(pred_n[i])
        pred_n[i]=0
print(len(temp))
# The answer is 4.
#Total no. of people in test set is 29.

#So now the accuracy of the set if we set everything to 0 in out prediction will be,
# from formula accuracy=true_potive/(true_positive+false_positive)
# so, accuracy= 25(which are true zeros)/(25(True zero)+4(false zero))=0.862
# with code following code can do same for you using accuracy_metrics function.
"""
temp=[]
for i in range(len(pred_n)):
    if pred_n[i]==1:
        temp.append(pred_n[i])
        pred_n[i]=0
#print(len(temp))
print(accuracy_score(labels_test,pred_n))  # should pring 0.862
"""

# so as we set every thing zero we don't have any true positive 1.


# Lets find both quantities with using precision_scoer(Accuracy) and recall_score(True positives)
from sklearn.metrics import precision_score
print(precision_score(labels_test,pred_n))
# the precision score is zero

# now for recall score
from sklearn.metrics import recall_score
print(recall_score(labels_test,pred_n))
# this is also zero.

