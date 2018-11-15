# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 6/07/2018

##This script converts train,test,validation set compatible with the (CNN) model. 
#usage: python CNN_format_conversion.py

import numpy
import pandas

#####################################################
# Convert validation set
dataframe = pandas.read_csv("Valid.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1364].astype(float)
Y = dataset[:,1364]

numpy.save('X_valid.np',X)
numpy.save('Y_valid.np',Y)

#####################################################
# Convert Training set
dataframe = pandas.read_csv("Train.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1364].astype(float)
Y = dataset[:,1364]

numpy.save('X_train.np',X)
numpy.save('Y_train.np',Y)

####################################################
# Convert Test set
dataframe = pandas.read_csv("Test.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1364].astype(float)
Y = dataset[:,1364]

numpy.save('X_test.np',X)
numpy.save('Y_test.np',Y)
