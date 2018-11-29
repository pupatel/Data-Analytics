
# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 07/08/2018

## Tokenization and Stopwords


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
from nltk.corpus import webtext,stopwords

#### SENTENCE TOKENIZE ######

para = "Hello World. It's good to see you. Thanks for buying this book."
print(sent_tokenize(para)) # output: ['Hello World.', "It's good to see you.", 'Thanks for buying this book.']

#### WORD TOKENIZE ######
sent= 'Hello World.'
print(word_tokenize(sent)) # output: ['Hello', 'World', '.']

#### ALTERNATIVE WORD TOKENIZER ######
para_1="Can't is a contraction."
tokenizer = WordPunctTokenizer() 
print(tokenizer.tokenize(para_1)) # output: ['Can', "'", 't', 'is', 'a', 'contraction', '.']

#### REGULAR EXPRESSION TOKENIZER ######
regex="Can't is a contraction."
tokenizer = RegexpTokenizer("[\w']+")
print (tokenizer.tokenize(regex)) # output: ["Can't", 'is', 'a', 'contraction']

#### TRAINING A SENTENCE TOKENIZER ######
text = webtext.raw('overheard.txt') # Read text example
sent_tokenizer = PunktSentenceTokenizer(text) # Train tokenizer on text
sents_tokenizer_1 = sent_tokenizer.tokenize(text) # Use new tokenizer
sents_tokenizer_2= sent_tokenize(text)  # Old tokenizer

#### FILTERING STOPWORDS ######

english_stops = set(stopwords.words('english')) #set english languagge and load stopwords
words = ["Can't", 'is', 'a', 'contraction']
print([word for word in words if word not in english_stops]) # output: ["Can't", 'contraction']

