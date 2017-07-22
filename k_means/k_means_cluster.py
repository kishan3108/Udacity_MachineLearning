#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit



def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)







### Find max and min value of 'exercised_stock_options' in data dictionary.
temp=[]
# The parameter you want to find
#param='exercised_stock_options'
param='salary'
for key,val in data_dict.items():
	if data_dict[key][param]=='NaN':
		pass
	else:
		temp.append(data_dict[key][param])
#now for max and min value from the list
print('max_val ',max(temp))
print('min_va ',min(temp))






# Last question whether scaling needed while comparing salary and from_messages
msg=[]
# The parameter you want to find
#param='from messages'
param='from_messages'
for key,val in data_dict.items():
	if data_dict[key][param]=='NaN':
		pass
	else:
		msg.append(data_dict[key][param])
print('max_val_from_msg',max(msg))
#14368
print('min_val_from_msg',min(msg))
#12
# As we can see the difference is too large the from message will only become a bias and there will no importance given to them.
# so it is critical to scale while looking in these two features.






### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
#for 2 features
features_list = [poi, feature_1, feature_2]
#for 3 features
#features_list = [poi, feature_1, feature_2,'total_payments']  #add 'total_payments' for 3rd parameter
data = featureFormat(data_dict, features_list )
poi,finance_features = targetFeatureSplit( data )
salary=(data[:,1])
exercised_stock_options=(data[:,2])






## MinMaxScaler function and tranform values for them for specific values.
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
sal_fit=scale.fit_transform(salary)
print(scale.transform([200000.]))
# 0.179976
stk_fit=scale.fit_transform(exercised_stock_options)
print(scale.transform([1000000.]))
# 0.02911 
#print(data)





### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)

#for 2 features uncomment below line
#for f1, f2  in finance_features:
#for 3 features uncomment below line
# added f3 as 3rd parameter
for f1, f2  in finance_features: 
    plt.scatter( f1, f2 )
plt.show()





### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
filt=KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred=filt.predict(finance_features)
# pred will give the clustered data.




### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters1.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"



