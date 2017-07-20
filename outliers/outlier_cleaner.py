#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    
    cleaned_data = []
    import numpy as np
    ## your code goes here
    errors=net_worths-predictions
    threshold=np.percentile(np.absolute(errors),90)
    temp=[]
    cleaned_data=[(age,net_worth,error) for age, net_worth, error in zip(ages,net_worths, errors) if abs(error) <= threshold]
    return cleaned_data

