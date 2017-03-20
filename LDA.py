
import os
import csv
import pandas as pd
from pattern.es import lemma
from gensim import corpora, models, similarities
from collections import defaultdict
import logging
import scipy.sparse as sp
import numpy as np
from itertools import chain
flatten = chain.from_iterable
from nltk import word_tokenize
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
import itertools
import pyLDAvis
import pandas as pd
import re
import simplejson


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = []

with open('/Users/roysourish/Desktop/SENG 607/project/batphone/0.06.RC1.csv', 'rb') as csvfile:
    df = pd.read_csv(csvfile)
    names = df['commit message']

#We start with documents represented as strings
for line in names:
    line=line.strip('\n')
    documents.append(line)

#print documents
stoplist = set('for a of the and to in is on &'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint  # pretty-printer
#pprint(texts)


dictionary = corpora.Dictionary(texts)
#dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
#print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
print corpus
'''
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
'''

lda = models.LdaModel(corpus, id2word=dictionary, num_topics= 50)
corpus_lsi = lda[corpus]
lda.print_topics(50)

for doc in corpus_lsi:
    print("********************DOCUMENTS*****************" ,doc)


prepared = pyLDAvis.prepare(**corpus_lsi.pyldavis_data())
pyLDAvis.display(prepared)

#for i in range(0, lda.num_topics-1):
#    print lda.print_topic(i)

#for doc in corpus_lsi: # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#    print(doc)


'''
new_vec = dictionary.doc2bow(documents.lower().split())
print(new_vec)


corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/Users/roysourish/Desktop/SENG 607/batphone0.01.mm', corpus)  # store to disk, for later use
print(corpus)

class MyCorpus(object):
    def __iter__(self):
        for line in open('/Users/roysourish/Desktop/SENG 607/project/batphone/0.01.csv'):
# assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()
print (corpus_memory_friendly)

#vec=[]
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)

#vec=vector

tfidf = models.TfidfModel(corpus)
#print(tfidf[doc2bow])

if (os.path.exists("/tmp/deerwester.dict")):
    dictionary = corpora.Dictionary.load('/Users/roysourish/Desktop/SENG 607/project/batphone/0.01.csv')
    corpus = corpora.MmCorpus('/Users/roysourish/Desktop/SENG 607/batphone0.01.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
documents.append(names)
print documents
with open('/Users/roysourish/Desktop/SENG 607/project/wirebug/wirebug-0.1.2.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        documents.append(row)

print documents
if (os.path.exists("/SENG 607/project/bitcoin-wallet/v2.46.csv")):
    dictionary = corpora.Dictionary.load('/SENG 607/project/bitcoin-wallet/v2.46.csv')
    corpus = corpora.MmCorpus('/SENG 607/project/bitcoin-wallet/v2.46.mm')
    print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")
    documents = ["Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"]
'''