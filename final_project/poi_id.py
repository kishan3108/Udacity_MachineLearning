#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','from_poi_to_this_person','long_term_incentive',
                 'salary_bonus_ratio','messages_to_poi_percentage' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)




# Following dictionaries is made to track the items and finding the max. and min. elements
"""
##Poi dictionary for future reference
poi_d={}
for key,val in data_dict.items():
    if data_dict[key]['poi']==True:
        poi_d.update({key:data_dict[key]['poi']})

# Salary dictionay for future reference
salary_d={}
for key,val in data_dict.items():
    if data_dict[key]['salary']!='NaN':
        salary_d.update({key:data_dict[key]['salary']})
            
#from_poi_this_person dictionaty for future reference
from_poi_d={}
for key,val in data_dict.items():
    if data_dict[key]['from_poi_to_this_person']!='NaN':
        from_poi_d.update({key:data_dict[key]['from_poi_to_this_person']})

#from_this_person_to_poi dictionary for future reference
long_term_d={}
for key,val in data_dict.items():
    if data_dict[key]['long_term_incentive' ]!='NaN':
        long_term_d.update({key:data_dict[key]['long_term_incentive' ]})

"""





        
### Task 2: Remove outliers
# Now as we looks at the max. elements in the features, we can find that there is 'TOTAL' element in data_dict which
# we learned in the class stands for 'TOTAL' salary which is  typo.
# So let's remove 'TOTAL' from data_dict.
#Following other people are checked who have several big parameters and are really not helping in prediction, So they are tried to being removed to optimize results.
#print(data_dict['TOTAL'])
data_dict.pop('TOTAL')
#print data_dict["THE TRAVEL AGENCY IN THE PARK"]
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
#data_dict.pop('LAVORATO JOHN J')
#data_dict.pop('DIETRICH JANET R')
data_dict.pop('MARTIN AMANDA K')





# These dictionaries are modification of previous dictionaries to track them after removing outliers.
"""
#from_poi_this_person dictionaty for future reference
from_poi_d={}
for key,val in data_dict.items():
    if data_dict[key]['from_poi_to_this_person']!='NaN':
        from_poi_d.update({key:data_dict[key]['from_poi_to_this_person']})

#from_this_person_to_poi dictionary for future reference
long_term_d={}
for key,val in data_dict.items():
    if data_dict[key]['long_term_incentive' ]!='NaN':
        long_term_d.update({key:data_dict[key]['long_term_incentive' ]})

"""







### Task 3: Create new feature(s)

#Following new features  are created to analyse the data.
#Lets create some new features like salary to bonus ratio
for name,features in data_dict.items():
    if features['bonus']!='NaN' and features['salary']!='NaN':
        features['salary_bonus_ratio']=float(features['salary'])/float(features['bonus'])
    else:
        features['salary_bonus_ratio']='NaN'

# Lets create another feature with ration of messages_to_poi_percentage
for name,features in data_dict.items():
    if features['from_messages']!='NaN' and features['from_this_person_to_poi']!='NaN':
        features['messages_to_poi_percentage']=float(features['from_this_person_to_poi'])/float(features['from_messages'])
    else:
        features['messages_to_poi_percentage']='NaN'








### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##Lets scale our feature first to give them all equal weitage
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
features=scale.fit_transform(features)


##Lets pick 2 best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
filt = SelectKBest(chi2, k=3)
features=filt.fit_transform(features,labels)
print(filt.scores_)
print(filt.get_support())








#lets devide total labels and features in to 70% train features and 30% for testing
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=42)









### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.metrics import accuracy_score
# Provided to give you a starting point. Try a variety of classifiers.
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
#without tuning the accuracy is 0.7575
"""
"""
from sklearn.ensemble import AdaBoostClassifier
clf=AdaBoostClassifier(n_estimators=50,learning_rate=1.0,algorithm='SAMME.R')
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
#without tuning the accuracy is 0.7575
"""

"""
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
#without tuning the accuracy is 0.7878
"""


# Lets try Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
#"""criterion='entropy',max_depth=6,min_samples_split=7,presort=True,max_leaf_nodes=None,max_features=3,splitter='best',min_samples_leaf=5"""
clf=DecisionTreeClassifier(criterion='entropy',max_depth=6,min_samples_split=7,presort=True,max_features=3,min_samples_leaf=5)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
#print(clf.feature_importances_)
#without tuning the accuracy is 0.72


"""
#Lets try SVM classifier
from sklearn import svm
clf=svm.SVC()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
# without tuning the accuracy is 0.90
#But the precision is failing every time so we are using decision tree classifier instead.
"""







### Task 5: Tune your classifier to achieve better than .3 precision and recall
# The tuning is been done in above part.
# I tried GridSearchCV using following range of parameters but the preceision and recall falls very short of needed 0.3.
# So above tuning is kept.

"""
from sklearn.model_selection import GridSearchCV
parameters={'max_depth':[1,10],'min_samples_split':[2,25],'max_features':[1,3]}
clf=GridSearchCV(clf, parameters)
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print(accuracy_score(labels_test,pred))
"""



from sklearn.metrics import recall_score,precision_score
print('Precision Score ',precision_score(labels_test,pred))
print('Recall Score ',recall_score(labels_test,pred))





### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
