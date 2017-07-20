#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
import numpy as np
print(np.shape(data))
data=np.delete(data,[67],0)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary,bonus)
plt.show()
#print(data['TOTAL'])
# Biggest outliers 'TOTAL'
# Its just a spreadshit quirk

#It has four more outliers

#Two most outliers are: LAY KENNETH L   and   SKILLING JEFFREY K


#Leave hem as they are also important...
