
import os
import csv
import pandas as pd
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
#from topik.run import run_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = []


with open('/Users/roysourish/Desktop/SENG 607/project/batphone/0.01.csv', 'rb') as csvfile:
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
stoplist.update(('and','I','A','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','11','+','->','>','<','2','3','4','5','6','7','8','9','0','1','usher','+', '-', '<=', '>=','http://translatewiki.net.','0.1','0.2','0.5'))

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
n_topics=5
lda = LdaModel(corpus_tfidf, id2word=dictionary, iterations =50, num_topics= n_topics)
#lda.save('/Users/roysourish/Desktop/SENG 607/corpus_apps/batphone_0.01.model')

#plt.plot(lda,n_topics)
#plt.show()

#run_model('data.json', field='abstract', model='lda_online', r_ldavis=True, output_file=True)


#names.to_csv(r'/Users/roysourish/Desktop/SENG 607/np.txt', header=None, index=None, sep=' ', mode='a')
## word lists
for i in range(0, n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term)
    print ("Top 10 terms for topic #" +str(i) + ": "+ ", ".join(str(i[0]) for i in terms))



#vis = pyLDAvis.gensim.prepare(**corpus_lsi)
#pyLDAvis.display(vis)

'''
def gensim_output(modelfile, corpusfile, dictionaryfile):
    """Displaying gensim topic models"""
    ## Load files from "gensim_modeling"
    corpus = corpora.MmCorpus(corpusfile)
    dictionary = corpora.Dictionary.load(dictionaryfile) # for pyLDAvis
    myldamodel = models.ldamodel.LdaModel.load(modelfile)
    
    ## Interactive visualisation
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(vis)
'''

###topic-words vectors: topics vs. words
from sklearn.feature_extraction import DictVectorizer

def topics_to_vectorspace(n_topics, n_words=100):
    rows = []
    for i in xrange(n_topics):
        temp = lda.show_topic(i, n_words)
        row = dict(((i[1],i[0]) for i in temp))
        rows.append(row)
    
    return rows

vec = DictVectorizer()

X = vec.fit_transform(topics_to_vectorspace(n_topics))
X.shape
#print X



pca = PCA(n_components=2)



## PCA
X_pca = pca.fit(X.toarray()).transform(X.toarray())

#plt.figure()
for i in xrange(X_pca.shape[0]):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
    plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + str(i))

plt.title('PCA Topics for the commit messages of an app batphone')
#plt.savefig("pca_topic")
#plt.xlim([-1,1])
#plt.ylim([-1,1])
plt.xlabel('eigen values')
plt.ylabel('eigen vectors')
plt.legend()

#plt.show()

X_pca = pca.fit(X.T.toarray()).transform(X.T.toarray())

plt.figure()
for i, n in enumerate(vec.get_feature_names()):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], alpha=.5)
    plt.text(X_pca[i, 0], X_pca[i, 1], s=' ' + n, fontsize=8)

plt.title('PCA Words of an app BARTs commit messages')
#plt.savefig("pca_words")
plt.show()

'''

## hierarchical clutering
X_pca = pca.fit(X.toarray()).transform(X.toarray())
plt.figure(figsize=(12,6))
Z=linkage(X_pca)
dendrogram(Z)
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=8)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=8)
#plt.savefig("dendro")
plt.title('Dendrogram')
#plt.show()


## correlation matrix
cor = squareform(pdist(X.toarray(), metric="euclidean"))
plt.figure(figsize=(12,6))
Z=linkage(cor)
dendrogram(Z)
ax = plt.gca()
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=8)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
dendrogram(Z, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=8)
#plt.savefig("dendro")
plt.title('Correlation Matrix')
#plt.show()



## network
pca_norm = make_pipeline(PCA(n_components=2), Normalizer(copy=False))

X_pca_norm = pca_norm.fit(X.toarray()).transform(X.toarray())

cor = squareform(pdist(X_pca_norm, metric="euclidean"))

G = nx.Graph()

for i in xrange(cor.shape[0]):
    for j in xrange(cor.shape[1]):
        if i == j:
            G.add_edge(i, j, {"weight":0})
        else:
            G.add_edge(i, j, {"weight":1.0/cor[i,j]})

edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > .8]
edge_weight=dict([((u,v,),int(d['weight'])) for u,v,d in G.edges(data=True)])

#pos = nx.graphviz_layout(G, prog="twopi") # twopi, neato, circo
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=350, alpha=.5)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

plt.savefig("network")
plt.show()
'''
