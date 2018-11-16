# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 07/08/2018

import nltk
from nltk.chunk import ne_chunk
from nltk.chunk import ChunkParserI
from nltk.chunk.util import conlltags2tree
from nltk.corpus import gazetteers

##  Extracting Common entity tags include PERSON, ORGANIZATION, and LOCATION

#### PART OF SPEECH TAGGING ######

#### EXTRACT NAMED ENTITY ######

tree = ne_chunk(treebank_chunk.tagged_sents()[0]) #chunk a single sentence into a Tree

def sub_leaves(tree, label): # This function gets the leaves of all the subtrees storing PERSON & ORGANIZATION
 return [t.leaves() for t in tree.subtrees(lambda s: label() == label)]

sub_leaves(tree, 'PERSON') # GET  PERSON 
sub_leaves(tree, 'ORGANIZATION') # GET ORGANIZATION

 print ([sub_leaves(t, 'ORGANIZATION') for t in trees])
 
####  EXRACTING PROPER NOUN CHUNKS ######
 chunker = RegexpParser(r'''... NAME:... {<NNP>+}... ''')  # Combines all proper nouns into a NAME chunk
 print sub_leaves(chunker.parse(treebank_chunk.tagged_sents()[0]),'NAME')

####  EXRACTING LOCATIONS ######
from chunkers import LocationChunker
from nltk.chunk.util import conlltags2tree
         
 def parse(self, tagged_sent): # This function parse sentence and identifies locations
  iobs = self.iob_locations(tagged_sent)
  return conlltags2tree(iobs)

t = loc.parse([('San', 'NNP'), ('Francisco', 'NNP'), ('CA','NNP'), ('is', 'BE'), ('cold', 'JJ'), ('compared', 'VBD'), ('to','TO'), ('San', 'NNP'), ('Jose', 'NNP'), ('CA', 'NNP')])
print (sub_leaves(t, 'LOCATION'))
