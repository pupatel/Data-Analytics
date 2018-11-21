# -*- coding: utf-8 -*-

# Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
# Date created: 07/08/2018

usage: python3 NLTK_SpaCy_Named_Entity_Recognition.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from bs4 import BeautifulSoup
import requests
import re


######   NAMED-ENTITY-RECOGNITION (NER) USING NLTK ###### 

## EXAMPLE CORPUS TEXT
example = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

## FUNCTION TO TOKENIZE AND PARTS-OF-SPEECHTAG
def preprocess(sentence):
    sent = nltk.word_tokenize(sentence)
    sent = nltk.pos_tag(sentence)
    return sentence
sentence = preprocess(example)
print (sentence)

## DEFINE AND BUILD NOUNCHUNK

# PATTERN FOR IDENTIFYING NOUN PHRASE: DETERMINER, DT -> ANY NUMBER OF ADJECTIVES, JJ -> NOUN, NN
pattern = 'NP: {<DT>?<JJ>*<NN>}'

# CREATE A CHUNK PARSER
chunk_parse = nltk.RegexpParser(pattern)
chunk_sent = chunk_parse.parse(sentence)
print(chunk_sent)

# GET CHUNK STRUCTURE
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

# RENDER CHUNK TREE STRUCUTRE
ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)


######  NAMED-ENTITY-RECOGNITION (NER) USING SPACY  ###### 


#LOAD SPACY NER MODEL
nlp = en_core_web_sm.load()

# APPLY NLP ONCE, WHOLE PIPLELINE RETURNS THE OBJECTS
document = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in document.ents])


## NER IN NYTIMES ARTICLE

# FUNCTIONS TO DOWNLOAD ARTICLE FROM NYTIMES

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

# APPLY SPACY NLP MODULE ON NYTIMES TEXT
nytimes_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(nytimes_bb)
print (len(article.ents))

# COUNTS OF UNIQUE ENTITY IN ARTICLE
labels = [x.label_ for x in article.ents]
print(Counter(labels))

# MOST FREQUENT 5 TOKENS IN ARTICLE
items = [x.text for x in article.ents]
Counter(items).most_common(5)

# FIND RANDOM SENTENCE FROM ARTICLE
sentences = [x for x in article.sents]
print(sentences[11])

# GENERATE THE RAW MARKUP AND HIGHLIGHT ENTITY
displacy.render(nlp(str(sentences[11])), jupyter=False, style='ent')

sys.exit()



