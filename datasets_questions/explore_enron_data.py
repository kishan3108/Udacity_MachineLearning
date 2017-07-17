#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
import numpy as np
print((enron_data['LAY KENNETH L']))
print()
print((enron_data['SKILLING JEFFREY K']))
print()
print((enron_data['FASTOW ANDREW S']))
print(len(enron_data))
temp=[]

for key,val in (enron_data.items()):
    if enron_data[key]['total_payments']=='NaN': #and enron_data[key]['poi']==True:
        pass
    else:
        temp.append(enron_data[key])
