# Obtained from NLTK 3 book by Jacob perkins

class LocationChunker(ChunkParserI):

 def __init__(self):
  self.locations = set(gazetteers.words())
  self.lookahead = 0
  for loc in self.locations:
    nwords = loc.count(' ')
    
    if nwords > self.lookahead:
    self.lookahead = nwords
    
    
 def iob_locations(self, tagged_sent):
  i = 0
  l = len(tagged_sent)
  inside = False
  while i < l:
    word, tag = tagged_sent[i]
      j = i + 1
      k = j + self.lookahead
      nextwords, nexttags = [], []
      loc = False
      while j < k:
       if ' '.join([word] + nextwords) in self.locations:
          if inside:
            yield word, tag, 'I-LOCATION'
          else:
            yield word, tag, 'B-LOCATION'
          for nword, ntag in zip(nextwords, nexttags):
            yield nword, ntag, 'I-LOCATION'
          loc, inside = True, True
          i = j
          break
        if j < l:
          nextword, nexttag = tagged_sent[j]
          nextwords.append(nextword)
          nexttags.append(nexttag)
          j += 1
          else:
            break
            
        if not loc:
          inside = False
          i += 1
          yield word, tag, 'O'
