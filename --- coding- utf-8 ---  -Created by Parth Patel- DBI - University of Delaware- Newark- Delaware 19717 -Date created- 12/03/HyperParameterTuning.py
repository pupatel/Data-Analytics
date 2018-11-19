# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 12/03/2017 

##This script builds Random Forest classifier and perfroms grid search for parameter tuning or optimization.

#usage: python3 HyperParameterTuning.py

import os,sys
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

InputFile='Features.csv'
OutputFile="Performance_Report.txt"
FeatureSet = pd.read_csv(InputFile,sep=',')
FeatureSet.head()
header= FeatureSet.columns.tolist()
cols=header[:-1]
colsRes = ['class']
Features = FeatureSet.as_matrix(cols) #training array - put 1-1372 features here with thier heading.
Feature_labels = FeatureSet.as_matrix(colsRes) # training results- put class value here.
col,row=Feature_labels.shape
Feature_labels = Feature_labels.reshape(col,)  
X,y= Features,Feature_labels
print("Step1: Dataset is loaded")


clf = RandomForestClassifier(criterion= "entropy")

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# use a full grid over all parameters
param_grid = {"max_depth": [3,4,5,6,7,8,None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "n_estimators":[10,20,50,100,150,200]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

sys.exit()

#----------------------------------------------------------------------------

