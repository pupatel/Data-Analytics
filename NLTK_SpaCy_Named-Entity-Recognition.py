import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint


ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
print (sent)


#nounchunk

pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)


#print chunk strucure
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)


ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)



# USE SPACY


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

# Apply nlp once, the entire background pipeline will return the objects
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])



#Real Examples of Named Entity recognition in NYTimes article


from bs4 import BeautifulSoup
import requests
import re

#Function to donwload article from NYTimes

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))

#Apply SPACY NLP MODULE ON NYTIMES TEXT
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)
print (len(article.ents))

#COUNTS OF UNIQUE ENTITY
labels = [x.label_ for x in article.ents]
print(Counter(labels))

#MOST FREQUENT 5 TOKENS
items = [x.text for x in article.ents]
Counter(items).most_common(5)


#FIND RANDOM SENTENCE
sentences = [x for x in article.sents]
print(sentences[11])

#GENERATE THE RAW MARKUP
displacy.render(nlp(str(sentences[11])), jupyter=False, style='ent')



