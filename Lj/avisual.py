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


with open('/Users/roysourish/Desktop/SENG 607/project/iFixitAndroid/v1.0.csv', 'rb') as csvfile:
    df = pd.read_csv(csvfile)
    names = df['commit message']

##To write the o/p to a text file
#names.to_csv(r'/Users/roysourish/Desktop/SENG 607/np.txt', header=None, index=None, sep=' ', mode='a')


#We start with documents represented as strings
for line in names:
    line=line.strip('\n')
    documents.append(line)

print(documents.shape)

