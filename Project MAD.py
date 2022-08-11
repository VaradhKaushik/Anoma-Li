# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:50:25 2022

@author: 212701595
"""

class MAD:
    from functools import reduce


def mean1(X):
    """
    This function returns the mean of the input dataframe. If the dataframe
    that is passed to this function is of length 0, an error will be raised
    that the dataframe was empty.
    param X: an input list of some length greater than 0.
    returns: the mean of the input list
    """
    if len(X) == 0:
        return "error: empty list - please define a dataframe"
    else:
        return reduce((lambda x, y: x + y), X) / len(X)

def abso(k):
    """
    This function calculates the absolute value of each value within the
    input list in order to prepare it for the calculation in subsequent
    functions
    param k: each value - mean within the input list X
    returns: the absolute value of k
    """
    if k is None:
        return "error: model has null values"
    else:
        return k if k >= 0 else -k


def mad(X):
    """
    This function calculates the mean absolute difference of values within 
    a given input list. For example, in an input list of X = 1,2,3 the mean
    absolute difference would be 0.66, which means that the absolute deviation
    of all data points from the mean is equal to 0.66. 
    param X: an input list of some length greater than 0.
    returns: the mean average deviation of the input list X
    """
    mean = mean1(X)
    difference = 0
    if len(X) == 0:
        return "error: empty list - please define a dataframe"
    for i in X:
        difference = difference + abso(i - mean)
    return difference/len(X)