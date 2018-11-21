# -*- coding: utf-8 -*-

# Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
# Date created: 08/17/2018

usage: python3 ML_Spacy_NLP_Text_classification.py
  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import spacy

stopwords = stopwords.words('english')
df = pd.read_csv('research_paper.csv')
df.head()

# SPLIT DATA INTO TRAIN AND TEST SETS
train_set, test_set = train_test_split(df, test_size=0.33, random_state=42)
print('Research title sample:', train_set['Title'].iloc[0])
print('Conference of this paper:', train_set['Conference'].iloc[0])
print('Training Data Shape:', train_set.shape)
print('Testing Data Shape:', test_set.shape)

# GROUPING & PLOTING RESERACH PAPERS BY DIFFERENCE CONFERENCES
fig = plt.figure(figsize=(8,5))
sns.barplot(x = train_set['Conference'].unique(), y=train_set['Conference'].value_counts())
plt.show()


#LOAD RESEARCH PAPERS INTO SPACY'S NLP MODULE AND PROCESSES IT
nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation

# CLEAN TEXT
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']
        tokens = [token for token in tokens if token not in stopwords and token not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
  
#LOAD TEXTS FROM INFOCOM AND ISCAS CONFERENCES  
INFOCOM_text = [text for text in train[train['Conference'] == 'INFOCOM']['Title']]
ISCAS_text = [text for text in train[train['Conference'] == 'ISCAS']['Title']]

# CLEAN TEXTS FROM INFOCOM AND ISCAS CONFERENCES 
INFOCOM_clean = cleanup_text(INFOCOM_text)
INFOCOM_clean = ' '.join(INFOCOM_clean).split()
ISCAS_clean = cleanup_text(ISCAS_text)
ISCAS_clean = ' '.join(ISCAS_clean).split()

# COUNT WORDS USED IN INFOCOM AND ISCAS PAPERS
INFOCOM_counts = Counter(INFOCOM_clean)
ISCAS_counts = Counter(ISCAS_clean)

# COUNT MOST WORDS USED IN INFOCOM AND ISCAS PAPERS
INFOCOM_common_words = [word[0] for word in INFOCOM_counts.most_common(20)]
INFOCOM_common_counts = [word[1] for word in INFOCOM_counts.most_common(20)]
fig2 = plt.figure(figsize=(18,6))
sns.barplot(x=INFOCOM_common_words, y=INFOCOM_common_counts)
plt.title('Most Common words used in the research papers for conference INFOCOM')
plt.show()

ISCAS_common_words = [word[0] for word in ISCAS_counts.most_common(20)]
ISCAS_common_counts = [word[1] for word in ISCAS_counts.most_common(20)]
fig3 = plt.figure(figsize=(18,6))
sns.barplot(x=ISCAS_common_words, y=ISCAS_common_counts)
plt.title('Most Common words used in the research papers for conference ISCAS')
plt.show()
