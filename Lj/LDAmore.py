
import logging
import os
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

MODELS_DIR = "/Users/roysourish/Desktop/SENG 607/corpus_apps"

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR,
                                                         "batphone_0.01.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "batphone_0.01.mm"))

tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "/Users/roysourish/Desktop/SENG 607/corpus_apps/coordsbat.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Next I clustered the points in the reduced 2D LSI space using KMeans, varying the number of clusters (K) from 1 to 10. The objective function used is the Inertia of the cluster, defined as the sum of squared differences of each point to its cluster centroid. This value is provided directly from the Scikit-Learn KMeans algorithm. Other popular functions include Distortion (Inertia divided by the number of points) or the Percentage of Variance Explained

MAX_K = 10

X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
'''
#I plotted the Inertias for different values of K, then used Vincent Granville's approach of calculating the third differential to find an elbow point. The elbow point happens here for K=5 and is marked with a red dot in the graph below.

#GENERATING CLUSTERS
X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
NUM_TOPICS=5
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_

colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
plt.show()
'''