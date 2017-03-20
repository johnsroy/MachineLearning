# -*- coding: utf-8 -*-
import os
import csv
import pandas as pd
import sys
from gensim import corpora, models, similarities
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from pattern.es import lemma
from gensim import corpora, models, similarities
from collections import defaultdict
import logging
import scipy.sparse as sp
import numpy as np
from numpy import array
from itertools import chain
flatten = chain.from_iterable
from nltk import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
import PIL
from os import path
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from math import factorial, cos, e
from scipy import *
from scipy.spatial.distance import pdist, squareform
import itertools
import pyLDAvis.gensim
import json
from nltk import word_tokenize
import warnings
import IPython
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import string

documents = []


with open('/Users/roysourish/Desktop/SENG 607/project/KindMind/Latest.csv', 'rb') as csvfile:
    df = pd.read_csv(csvfile)
    names = df['commit message']

##To write the o/p to a text file
#names.to_csv(r'/Users/roysourish/Desktop/SENG 607/np.txt', header=None, index=None, sep=' ', mode='a')


#We start with documents represented as strings
for line in names:
    line=line.strip('\n')
    documents.append(line)

#print documents

stoplist = set(stopwords.words('english'))
stoplist.update(('and','I','A','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','11','+','->','>','<','2','3','4','5','6','7','8','9','0','1','usher','+', '-', '<=', '>='))

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
# store the dictionary, for future reference
#dictionary.save('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphone_0.01.dict')
#print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
#Create Market Matrix
#corpora.MmCorpus.serialize('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphone_0.01.mm', corpus)
#print corpus
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
n_topics=2
lda = LdaModel(corpus_tfidf, id2word=dictionary, iterations =50, num_topics= n_topics)
#lda.save('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphone_0.01.model')

#plt.plot(lda,n_topics)
#plt.show()

#run_model('data.json', field='abstract', model='lda_online', r_ldavis=True, output_file=True)


#names.to_csv(r'/Users/roysourish/Desktop/SENG 607/np.txt', header=None, index=None, sep=' ', mode='a')
## word lists
for i in range(0, n_topics):
    temp = lda.show_topic(i, 5)
    terms = []
    for term in temp:
        terms.append(term)
    print ("Top 5 terms for topic #" +str(i) + ": "+ ", ".join(str(i[0]) for i in terms))


#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "maven-release-plugin], release, prepare, zxing-2.2, rollback, changes, c++, german, inspection, remove, issue, pdf417, add, test, remove, update, c++, port, issue, fix,issue, fix, add, use, remove"

vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lda[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lda[corpus])
index.save('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphoneandbart.index')
index = similarities.MatrixSimilarity.load('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphoneandbart.index')

sims = index[vec_lsi]
print(list(enumerate(sims)))

#sims = sorted(enumerate(sims), key=lambda item: -item[1])
#print(sims)

'''
dictionary = corpora.Dictionary.load('/Users/roysourish/Desktop/SENG 607/corpus_apps/BartLatest.dict')
corpus = corpora.MmCorpus("/Users/roysourish/Desktop/SENG 607/corpus_apps/BartLatest.mm")
lda = models.LdaModel.load("/Users/roysourish/Desktop/SENG 607/corpus_apps/BartLatest.lda") #result from running online lda (training)

index = similarities.MatrixSimilarity(lda[corpus])
index.save("simIndex.index")

docname = "/Users/roysourish/Desktop/SENG 607/project/batphone/0.01.csv"
doc = open(docname, 'r').read()
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow]

sims = index[vec_lda]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sims
'''