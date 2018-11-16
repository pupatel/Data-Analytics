# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 07/08/2018

##  lemmas and synonyms in WordNet

import nltk
from nltk.corpus import wordnet

#### GET SYNSETS ######
syns = wordnet.synsets("cookbook")
print("all synonymes:",syns)
print("First one:",syns[0].name())
print("First one - lemmas",syns[0].lemmas()[0])
print(syns[0].examples())

#### GET  SYNONYMS & ANTONYMS ######
synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
  for lemma in syn.lemmas():
    synonyms.append(lemma.name())
    if lemma.antonyms():
      antonyms.append(lemma.antonyms()[0].name())
      print(antonyms.append(lemma.antonyms()[0].name()))

print(set(synonyms))
print(set(antonyms))

#### CHECK WORD SIMILARITY  ######
ref= wordnet.synset('refernce.n.01')
bb = wordnet.synset('bibilography.n.01')
print(ref,bb)
print(ref.wup_similarity(bb)) ## Wu-Palmer semantic similarity
