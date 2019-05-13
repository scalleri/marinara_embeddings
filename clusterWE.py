import argparse
import gensim
from gensim.models import Word2Vec
import numpy as np
from collections import Counter
import random
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
import pickle

modelname = sys.argv[1]
clusternumber = int(sys.argv[2])

print('load model')
model = Word2Vec.load(modelname)

X = model[model.wv.vocab]


# bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=int(clusternumber))
# 
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# 
print('start clusteirng')
kmeans = cluster.MiniBatchKMeans(n_clusters=clusternumber,n_init=30,verbose=1) ## parameter genauer anschauen
kmeans.fit(X)



labels = kmeans.labels_
centroids = kmeans.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

words = list(model.wv.vocab)

f = open(modelname+".cluster_"+str(clusternumber)+".txt", "w")
for i, word in enumerate(words):
	center_word = model.most_similar(positive=[centroids[labels[i]]], topn=3)
	#print(center_word)
	f.write(word + "\t" + str(labels[i])+"\t"+str(center_word[0][0])+"\t"+str(center_word[0][1])+"\t"+str(center_word[1][0])+"\t"+str(center_word[1][1])+"\t"+str(center_word[2][0])+"\t"+str(center_word[2][1])+"\n")

with open(modelname+'kmeans_model.sav','w') as out_dump:
	pickle.dump(kmeans,out_dump)




# import matplotlib.pyplot as plt
# from itertools import cycle
# 
# plt.figure(1)
# plt.clf()
# 
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = centroids[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#              
# 
# #plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# 
