

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 07/09/2018 

# This script precdits readmission based on hospital discharge summary using Pandas, NLTK, and Sci-kit learn. 
# Three stages: (1) Prepare admission table each patiens (2) Prepare notes for patients (3) use Bag-of-words and ML to predict readmission date

### IMPORTANT NOTE: ADMISSION.csv AND NOTEEVENTS.csv are available freely to download from The MIMIC-III Clinical Database ###

#usage: python3 Readmission_EHR_NLTK_ML_Classififer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


########## STEP 1: PREPARE ADMISSION TABLE EACH PATIENT  ############

# LOAD ADMISSION FILE


# READ THE ADMISSIONS TABLE
df_admission = pd.read_csv('ADMISSIONS.csv')

## CHANGE DATES TO PROPER DATE FORMATE AND FLAG MISSING DATES  #####
df_admission.ADMITTIME = pd.to_datetime(df_admission.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admission.DISCHTIME = pd.to_datetime(df_admission.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_admission.DEATHTIME = pd.to_datetime(df_admission.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


# GET NEXT UNPLANNED ADMISSION DATE
# FIRST, SORT BY subject_ID and admission date #####
df_admission = df_admission.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admission = df_admission.reset_index(drop = True)


# SECOND, ADD THE NEXT ADMISSION DATE AND TYPE FOR EACH SUBJET USING GROUPBY ####
df_admission['NEXT_ADMITTIME'] = df_admission.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
# THIRD, GET THE NEXT ADMISSION TYPE
df_admission['NEXT_ADMISSION_TYPE'] = df_admission.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)



# FOURTH, GET ROWS WITH ADMISSION IS "ELECTIVE" AND REPLACE WITH "naT" OR "nan"
rows = df_admission.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_admission.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
df_admission.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN


# FIFTH, SORT AGAIN BY SUBJECT ID & ADMISSION TIME AND BACKFILL REMOVED ROWS WITH THE NEXT VALID ENTRY FOR NEXT ADMISSION DATE AND TYPE
df_admission = df_admission.sort_values(['SUBJECT_ID','ADMITTIME'])
df_admission[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_admission.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

#CALCULATE THE DAYS UNTIL NEXT ADMISSION
df_admission['DAYS_NEXT_ADMIT']=  (df_admission.NEXT_ADMITTIME - df_admission.DISCHTIME).dt.total_seconds()/(24*60*60)


########## STEP 2: PREPARE AND PARSE NOTES FOR EACH PATIENT   ############

df_notes = pd.read_csv("NOTEEVENTS.csv")

#FILTER ON DISCHARGE SUMMARY
df_notes_dis_sum = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']


#KEEP THE LAST DISCHARGE SUMMARY NOTE FROM MULTPLE NOTES
df_notes_dis_sum_last = (df_notes_dis_sum.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
assert df_notes_dis_sum_last.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'

#MERGE THE ADMISSION AND NOTES TABLES USING LEFT JOIN TO ACCOUNT FOR  MISSING NOTES

df_admission_notes = pd.merge(df_admission[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],df_notes_dis_sum_last[['SUBJECT_ID','HADM_ID','TEXT']], on = ['SUBJECT_ID','HADM_ID'],how = 'left')
assert len(df_admission) == len(df_admission_notes), 'Number of rows increased'

# CHECK TO SEE HOWMANY ADMISSION ARE MISSING NOTES 
#df_admission_notes.TEXT.isnull().sum() / len(df_admission_notes)


# CHECK TO SEE HOWMANY ADMISSION ARE MISSING NOTES GROUP BY EACH ADMISSION TYPE 
#df_admission_notes.groupby('ADMISSION_TYPE').apply(lambda g: g.TEXT.isnull().sum())/df_admission_notes.groupby('ADMISSION_TYPE').size()

# ADD A CLASSIFICATION OUTPUT LABELS FOR READMISSION: 1 = readmitted (< 30 DAYS), 0 = not readmitted (> 30 DAYS)
df_admission_notes_clean['OUTPUT_LABEL'] = (df_admission_notes_clean.DAYS_NEXT_ADMIT < 30).astype('int')

# SHUFFLE SAMPLES
df_admission_notes_clean = df_admission_notes_clean.sample(n = len(df_admission_notes_clean), random_state = 42)
df_admission_notes_clean = df_admission_notes_clean.reset_index(drop = True)

# SAVE 30% OF THE DATA AS VALIDATION AND TEST 
df_valid_test=df_admission_notes_clean.sample(frac=0.30,random_state=42)
df_test = df_valid_test.sample(frac = 0.5, random_state = 42)
df_valid = df_valid_test.drop(df_test.index)

# USE REST FOR TRAINING
df_train_all=df_admission_notes_clean.drop(df_valid_test.index)

## PREPARE TRAINING DATA ##

# SPLIT TRAINING DATA INTO POSIVIE AND NEGATIVE
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# USE UNDERSAMPLING TO BALANCE NEGATIVE SET
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

# SHUFFLE TRAINING SET
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

########## STEP 3: USE BAG-OF-OWRDS AND MACHINE LEARNING FOR PREDICTION ############

## USING BAG-OF-WORDS TECHNIQUE TO PARSE NOTES ##

# FIRST FILL EMPTY NOTES WITH SPACE AND REMOVE NEWLINES & CARRIAGE RETURNS

def preprocess_text(df):
    df.TEXT = df.TEXT.fillna(' ')
    df.TEXT = df.TEXT.str.replace('\n',' ')
    df.TEXT = df.TEXT.str.replace('\r',' ')
    return df

# PREPROCESS TRAINING,VALIDATION,TEST SETS
df_train = preprocess_text(df_train)
df_valid = preprocess_text(df_valid)
df_test = preprocess_text(df_test)


def tokenizer_better(text):
    # tokenize the text by replacing punctuation and numbers with spaces and lowercase all words
    
    punc_list = string.punctuation+'0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens

#FILTER STOPWORDS FROM THE LIST OF NLTK OR USER DEFINED LIST
my_stop_words = set(stopwords.words('english'))
#my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name','is','patient','s','he','at','as','or','one','she','his','her','am','were','you','pt','pm','by','be','had','your','this','date',                'from','there','an','that','p','are','have','has','h','but','o',                'namepattern','which','every','also','should','if','it','been','who','during', 'x']

#BUILD COUNTVECTOR FOR EACH WORD IN THE NOTE
word_vect = CountVectorizer(max_features = 3000, tokenizer = tokenizer_better, stop_words = my_stop_words)

# FIT (TRAIN) THE VECTONIZER
word_vect.fit(df_train.TEXT.values)
# X = word_vect.transform(sample_text)
# print (X.toarray())
# print (word_vect.get_feature_names())

# TRANFORM TRAIN AND VALIDATION SETS INTO NUMERICAL VALUES
X_train_tf = word_vect.transform(df_train.TEXT.values)
X_valid_tf = word_vect.transform(df_valid.TEXT.values)
X_test_tf = word_vect.transform(df_test.TEXT.values)

# SET OUTPUT LABELS
y_train = df_train.OUTPUT_LABEL
y_valid = df_valid.OUTPUT_LABEL
y_test = df_test.OUTPUT_LABEL

##BUILD A MACHINE LEARNING MODEL##

# LOGISITC REGRESSION
clf=LogisticRegression(C = 0.0001, penalty = 'l2', random_state = 42)
clf.fit(X_train_tf, y_train)

#SAVE MODEL
model = clf
y_train_preds = model.predict_proba(X_train_tf)[:,1]
y_valid_preds = model.predict_proba(X_valid_tf)[:,1]
y_test_preds = model.predict_proba(X_test_tf)[:,1]

print(y_train_preds)
print(y_valid_preds)
print(y_test_preds)

print ("Done")
