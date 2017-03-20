import gensim
import pandas as pd
from pattern.es import lemma
from gensim import corpora, models, similarities
import pyLDAvis.gensim
import IPython
import csv
import json
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from collections import defaultdict
import IPython
import os
import glob
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

MODELS_DIR = "/Users/roysourish/Desktop/SENG 607/corpus_apps"

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR,
                                                         "BartLatest.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "BartLatest.mm"))

tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()
'''
followers_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(followers_data)

lda = models.LdaModel.load('/Users/roysourish/Desktop/SENG 607/corpus_apps/BartLatest.lda')
followers_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(followers_data)
'''

'''
######### writes csv files to json #################
 
csvfile = open('/Users/roysourish/Desktop/SENG 607/project/BART/Latest.csv', 'r')
jsonfile = open('/Users/roysourish/Desktop/SENG 607/project/BART/Latest.json', 'w')

fieldnames = ("commit id","commit date","commit message")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    if row==0:
        pass
    else:
        json.dump(row, jsonfile)
        jsonfile.write('\n')




def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)

path = '/Users/roysourish/Desktop/SENG 607/project/BART'
fmask = os.path.join(path, '.csv')
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            f=open(file, 'r')
            df = pd.read_csv(csvfile)
            names = df['commit message']
                #  perform calculation
            f.close()
print names
'''
