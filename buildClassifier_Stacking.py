# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 12/03/2017 

##This script builds Ensemble of classifiers using a technique called "stacking" and perfroms stratified 5 fold CV five complete times and outputs accuracy, presicion, recall, roc curve, and AUC.

import os,sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from scipy import interp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection

from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
#from sklearn.feature_selection import SelectKBest,f_classif,SelectFromModel
#from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
#from sklearn import metrics
#from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_curve,auc
#from sklearn.tree import export_graphviz
#from sklearn import tree
import pandas as pd
import numpy as np
#from numpy  import array
import matplotlib.pyplot as plt
#from matplotlib import pyplot
#import pydotplus
#import pydot
#from IPython.display import Image 
#from sklearn.externals.six import StringIO
#from sklearn import tree
#import pylab as pl
if sys.version < '2.6':
    print ('You are using a version of Python that this program does not support. Please update to the latest version!')
    sys.exit(1)


# - - - - - G L O B A L   V A R I A B L E S   F O R   C L A S S I F I E R   - - - - - -
n_trees=100 # The number of trees in the forest
n_features=5 # The number of features to consider when looking for the best split
n_classes=2 # Binary class classification
n_folds=2 # Number of k-fold cross-validation
n_jobs= 10 # The number of jobs to run in parallel
Best_k_features=250 # Number of features to select using Information gain
# Prediction_Treshold=0.33 #0.35
#######################################################


# - - - - - G L O B A L   V A R I A B L E S   F O R   I N P U T    A N D    O U T P U T    F I L E S   - - - - - -
InputFile='Features.csv'
OutputFile="Performance_Report.txt"
FeatureSet = pd.read_csv(InputFile,sep=',')

#Set current directory 
current_directory= os.getcwd()

#Create a new directory called "Results"
new_directory=current_directory+"\Results"
if not os.path.exists(new_directory):
    os.mkdir(new_directory) 

# Now change old directory
os.chdir( new_directory )
OUT=open(OutputFile,"w") 
#######################################################

def RandomForest(n_trees,n_features,n_jobs):

    """ Initialze Random Forest classifier

    Args:
        n_trees: The number of trees in the forest
        n_features: The number of features to consider when looking for the best split
        n_jobs: The number of jobs to run in parallel

    Returns:
        The random forest classifier

    """

    rf = RandomForestClassifier(n_estimators=n_trees,criterion="entropy",max_features=n_features,n_jobs=n_jobs) # initialize
    return rf

def SVM():
    clf = svm.SVC(C=2,kernel='rbf',degree=3,random_state=1)
    return clf

def KNN():
    
    clf = KNeighborsClassifier(n_neighbors=5)
    return clf

def NB():
    clf=GaussianNB()
    return clf
    
def LR():
    
    lr = LogisticRegression()
    return lr

def Stacking(RF,SVM,KNN,NB,LR):
    
    sclf = StackingClassifier(classifiers=[RF, SVM, KNN, NB], meta_classifier=LR)
    return sclf
 


def selectKImportance(model, X, k):

    """ Select K Important features

    Args:
        model: Model under features are being selected
        X: Dataset
        k: Number of features to select 

    Returns:
        Dataset reduced to number of features wanted
        Location of each feature in the model
    """

    k_best_features_location=[]
    k_best_features_location=model.feature_importances_.argsort()[::-1][:k]
    return X[:,model.feature_importances_.argsort()[::-1][:k]],k_best_features_location
 
def FeatureSelction(n_trees,Features,Feature_labels,cols):

    """ Feature Selction using Inforamtion Gain

    Args:
        n_trees: The number of trees in the forest
        Features: Dataset with all features execpt last coloum which is class label
        Feature_labels: class labels
        cols: Total number of features
        
    Returns:
        Dataset reduced to number of features wanted
        Location of each feature in the model
        Top K important features ranked by thier Information Gain.
    """

    model = ExtraTreesClassifier(n_estimators=n_trees,n_jobs=2)
    model.fit(Features, Feature_labels)
    X_new,k_best_features_location = selectKImportance(model,Features,Best_k_features)
    important_features= sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), cols), reverse=True)
    print ("Step2: Feature Selection is perfomred")
    return X_new,k_best_features_location,important_features

    
def stdev(lst):

    """Calculates standard deviation
    Args:
        lst: List of numbers
    
    Returns: standard deviation for list of numbers
    """
    sum = 0
    mn = mean(lst)
    for i in range(len(lst)):
        sum += pow((lst[i]-mn),2)
    return sqrt(sum/len(lst)-1)
    

def cross_validaiton(FeatureSet,n_folds,n_trees,n_features,n_jobs,X_new,Feature_labels,clf):

    """ Perform k-fold cross validaiton k complete times

    Args:
        FeatureSet: Entire dataset
        n_folds: Number of k-fold cross-validation
        n_trees: The number of trees in the forest
        n_features: Number of randomly sampled features as candidates at each split.
        n_jobs: The number of jobs to run in parallel
        X_new: Dataset reduced to number of features wanted
        Feature_labels: class labels
        clf: Classifier to be used for cross-validation
                
    Returns:
        List of the following items for each cross-validation:
        Accuracy
        Standard Deviation of Accuracy
        Sensitivity
        Standard Deviation of Sensitivity
        Positive Predictive Value
        Standard Deviation of Positive Predictive Value
        Specificity
        and Feature label converted from "yes":1 and "no":0

    """

    #Map feature_labels from "yes":1 and "no":0
    Feature_labels_converted= FeatureSet['class'].tolist()
    Feature_labels_converted = [y.replace('yes', '1') for y in Feature_labels_converted]
    Feature_labels_converted = [y.replace('no', '0') for y in Feature_labels_converted]
    Feature_labels_converted=np.array(Feature_labels_converted,dtype=int)

    # Create empty lists
    ACC=[]
    ACC_stdev=[]
    RE=[]
    RE_stdev=[]
    SP=[]
    SP_stdev=np.array(n_folds,dtype=float)
    PPV=[]
    PPV_stdev=[]
 
    
    #Parth is trying a new ML appraoch
    clf1 = SVM()
    clf2 = clf#RandomForestClassifier(random_state=1)
    clf3 = KNN()
    clf4 =  NB()    
    lr = LR()
    sclf = Stacking(clf1,clf2,clf3,clf4,lr)
    
    for clf, label in zip([clf1, clf2, clf3,clf4, sclf], ['Support Vector Machine', 'Random Forest', 'KNN', 'Naive Bayes','StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X_new, Feature_labels_converted, cv=n_folds, scoring='accuracy')    
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    sys.exit()
    

    for i in range(n_folds): # Run five complete runs of 5-fold cross-valdation to ensure that the results does not depend on one 5-way split for one time 5-fold cross-validation.

        skf = StratifiedKFold(n_splits=n_folds,random_state=i, shuffle=True)      
        
        # Compute Accuracy
        accuracy = cross_val_score(sclf, X_new, Feature_labels_converted, cv=skf, scoring='accuracy')
        #accuracy = cross_val_score(clf, X_new, Feature_labels, cv=skf, scoring='accuracy')
        ACC.append(accuracy.mean())
        ACC_stdev.append(accuracy.std())
        
        # Compute Recall 
        recall = cross_val_score(sclf, X_new, Feature_labels_converted, cv=skf, scoring='recall')
        RE.append(recall.mean())
        RE_stdev.append(recall.std())

        # Compute Precision 
        precision = cross_val_score(sclf, X_new, Feature_labels_converted, cv=skf, scoring='precision')
        PPV.append(precision.mean())
        PPV_stdev.append(precision.std())
        
        # Compute Specificity 
        y_prediction=cross_val_predict(sclf,X_new, Feature_labels_converted,cv=skf)
        CM=confusion_matrix(Feature_labels_converted, y_prediction)
        specificity=float(CM[0][0])/float(CM[0][0]+CM[0][1])
        SP.append(specificity)
        
    print ("Step3: %s-fold cross validation is performed" % (n_folds)) 
    return ACC,ACC_stdev,RE,RE_stdev,PPV,PPV_stdev,SP,Feature_labels_converted


def stratified_cv(X, y, clf_class, shuffle=True, n_folds=n_folds, **kwargs):

    """ Perform stratified k-fold cross validaiton k complete times

    Args:
        X: Feautures
        y: class labels
        cls_class: Name of the classififer to be used
        n_folds: Number of k-fold cross-validation
        n_trees: The number of trees in the forest
        n_features: Number of randomly sampled features as candidates at each split

    Returns:
        y_pred: List of predcitions of a test set for each cross validation 
    """

    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


def ROC(n_folds,X_new,Feature_labels_coverted,clf):

    """ Plot ROC and compute AUC

    Args:
        n_folds: Number of k-fold cross-validation
        X_new: Dataset reduced to number of features wanted
        n_trees: The number of trees in the forest
        Feature_labels_coverted: Feature label converted from "yes":1 and "no":0
        clf: Classifier to be used for cross-validation

    Returns:
        List of the following items for each cross-validation:
        Average AUC
        Standard Deviation of AUC 
    """

    print ("----------------Computing ROC curve and AUC ---------------------")    
    #ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = StratifiedKFold(n_splits=n_folds,random_state=2,shuffle=True)   
    i = 0
    colors=['#E8175D','#D75C37','#2DCC70','#9A59B5','#34495E']
    plt.rc('grid', linestyle="--", color='#E8E8E8')
    plt.grid(True)

    for train, test in cv.split(X_new, Feature_labels_coverted):
        
        probas_ = clf.fit(X_new[train], Feature_labels_coverted[train]).predict_proba(X_new[test])
         # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Feature_labels_coverted[test], probas_[:, 1])
        #print "fpr: ", fpr,"\n tpr: ",tpr,"\n threshold: ",thresholds
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,color=colors[i],
                 label='ROC fold %d (AUC = %0.3f)' % (i+1, roc_auc))
        i += 1    
        
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='black',
             label='Random chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
    #         label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
             lw=1.5, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='red', alpha=.2,
                     label=r'$\pm$ 1 SD')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Detailed.png',dpi=600)

    return mean_auc,std_auc



def Write_results(ACC,ACC_stdev,RE,RE_stdev,PPV,PPV_stdev,SP,mean_auc,std_auc,important_features,Features,Feature_labels):

    """ Write Results into the OutputFile

    Args:
        List of the following items :
        Accuracy
        Standard Deviation of Accuracy
        Sensitivity
        Standard Deviation of Sensitivity
        Positive Predictive Value
        Standard Deviation of Positive Predictive Value
        Specificity
        Average AUC
        Standard Deviation of AUC
        Important Features
        Features
        and Feature labels 
    """

    print ("----------------Performance Measurements ---------------------")
    OUT.write("----------------Performance Measurements ---------------------\n")

    print ("Accuracy: %0.2f (+/- %0.2f)" % (sum(ACC)/len(ACC),sum(ACC_stdev)/len(ACC_stdev)))
    OUT.write("Accuracy: %0.2f (+/- %0.2f) \n" % (sum(ACC)/len(ACC),sum(ACC_stdev)/len(ACC_stdev)))

    print ("----------------RECALL----------------------")
    print ("Recall: %0.2f (+/- %0.2f)" % (sum(RE)/len(RE),sum(RE_stdev)/len(RE_stdev)))
    OUT.write("Recall: %0.2f (+/- %0.2f) \n" % (sum(RE)/len(RE),sum(RE_stdev)/len(RE_stdev)))

    print ("----------------POSITIVE PREDCITIVE VALUE ---------------------")
    print ("PPV: %0.2f (+/- %0.2f)" % (sum(PPV)/len(PPV),sum(PPV_stdev)/len(PPV_stdev)))
    OUT.write("PPV: %0.2f (+/- %0.2f) \n" % (sum(PPV)/len(PPV),sum(PPV_stdev)/len(PPV_stdev)))

    print ("----------------SPECIFICITY ---------------------")
    print ("SP: %0.2f (+/- %0.2f)" % (sum(SP)/len(SP),np.std(SP)))
    OUT.write("SP: %0.2f (+/- %0.2f)\n" % (sum(SP)/len(SP),np.std(SP)))

    print ("----------------AUC ---------------------")
    print ("AUC: %0.2f (+/- %0.2f)" % (mean_auc,std_auc))
    OUT.write("AUC: %0.2f (+/- %0.2f)\n" % (mean_auc,std_auc))

    y_pred=stratified_cv(Features,Feature_labels,RandomForestClassifier,n_estimators=n_trees,criterion="entropy",max_features=n_features,random_state=None)
    df_confusion = pd.crosstab(Feature_labels, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print ("----------------############################## -----------")

    print ("\n----------------Confusion Matrix ---------------------")
    print (df_confusion)
    OUT.write("\n----------------Confusion Matrix ---------------------\n %s \n" % (df_confusion))
    print ("----------------############################## -------")

    print ("----------------Classification Report ---------------------")
    OUT.write("\n----------------Classification Report ---------------------\n")
    classification_rep= classification_report(Feature_labels, y_pred)
    OUT.write(classification_rep)
    print (classification_rep)
    print ("----------------############################## ---------------------")

    OUT.write("\n----------------Feature Importance and Ranking ---------------------\n\n")
    print ("----------------Feature Importance and Ranking ---------------------")    
    OUT.write("InforamtionGain\tFeatures\n")

    i=1
    for feat in important_features:
        if (i<Best_k_features):
            OUT.write("%s\t%s\n" % (feat[0],feat[1]))
            
        i+=1
        
    print ("\nResults written")


def main(Features,Feature_labels,cols):

    """ Main function calling the aforementioned helper functions to execute all tasks

    Args:
    Features: Feature set
    Feature_labels: class labels
    cols: Number of features
    """ 

    #Step 2: Feature selection based on Information Gain
    X_new,k_best_features_location,important_features=FeatureSelction(n_trees,Features,Feature_labels,cols)                  

    #Step 3: Build Random Forest classifier
    clf= RandomForest(n_trees,n_features,n_jobs)
    
    #Step 4: Cross-validation experiment and calculating perfromance measurements
    ACC,ACC_stdev,RE,RE_stdev,PPV,PPV_stdev,SP,Feature_labels_converted=cross_validaiton(FeatureSet,n_folds,n_trees,n_features,n_jobs,X_new,Feature_labels,clf)
    
    #Plots ROC and calculates AUC
    mean_auc,std_auc=ROC(n_folds,X_new,Feature_labels_converted,clf)

    # Last Step: Write Results
    Write_results(ACC,ACC_stdev,RE,RE_stdev,PPV,PPV_stdev,SP,mean_auc,std_auc,important_features,Features,Feature_labels)

    # # Now change old directory
    os.chdir(current_directory )
    OUT.close()
  
    
if __name__ == '__main__':

    """ Calls Main function and exit upon completion"""
    
    #Step 1: Load dataset
    FeatureSet.head()
    header= FeatureSet.columns.tolist()
    cols=header[:-1]
    colsRes = ['class']
    Features = FeatureSet.as_matrix(cols) #training array - put 1-1372 features here with thier heading.
    Feature_labels = FeatureSet.as_matrix(colsRes) # training results- put class value here.
    col,row=Feature_labels.shape
    Feature_labels = Feature_labels.reshape(col,)    
    print("Step1: Dataset is loaded")

    #Call Main function
    main(Features,Feature_labels,cols)
    
    print ("Done")
    sys.exit()

    
