
# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 07/08/2018

## Tokenization and Stopwords


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer,PunktSentenceTokenizer
from nltk.corpus import webtext

#### SENTENCE TOKENIZE ######

para = "Hello World. It's good to see you. Thanks for buying this book."
print(sent_tokenize(para)) # output: ['Hello World.', "It's good to see you.", 'Thanks for buying this book.']

#### WORD TOKENIZE ######
sent= 'Hello World.'
print(word_tokenize(sent)) # output: ['Hello', 'World', '.']

#### ALTERNATIVE WORD TOKENIZER ######

tokenizer = WordPunctTokenizer() 
print(tokenizer.tokenize(para)) # output: ['Can', "'", 't', 'is', 'a', 'contraction', '.']

#### REGULAR EXPRESSION TOKENIZER ######
regex="Can't is a contraction."
tokenizer = RegexpTokenizer("[\w']+")
print (tokenizer.tokenize(regex)) # output: ["Can't", 'is', 'a', 'contraction']

#### Training a sentence tokenizer ######
text = webtext.raw('overheard.txt') # Read text example
sent_tokenizer = PunktSentenceTokenizer(text) # Train tokenizer
sents_tokenizer_1 = sent_tokenizer.tokenize(text) # Use new tokenizer
sents_tokenizer_2= sent_tokenize(text)  #Old tokenizer
