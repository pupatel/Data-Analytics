#Simple SpaCy tutorial from SpaCy's website

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

# Process whole documents
text = (u"When Sebastian Thrun started working on self-driving cars at "
        u"Google in 2007, few people outside of the company took him "
        u"seriously. “I can tell you very senior CEOs of major American "
        u"car companies would shake my hand and turn away because I wasn’t "
        u"worth talking to,” said Thrun, now the co-founder and CEO of "
        u"online higher education startup Udacity, in an interview with "
        u"Recode earlier this week.")
# nlp will apply all modules of natural language processing in single command        
document = nlp(text)

# Find named entities, phrases and concepts
for entity in document.ents:
    print(entity.text, entity.label_)

# Determine semantic similarities
document1 = nlp(u"my fries were super gross")
document2 = nlp(u"such disgusting fries")
similarity = document1.similarity(document2)
print(document1.text, document2.text, similarity)
