import csv, io, pickle, time, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter


start = time.time()

data = np.load('call_data.npy')

print(data.columns)

print('time to get data:', time.time()-start)

data = data[:10000]

#default is 8 clusters
kmeans = KMeans(n_clusters=20, max_iter=20)
labels = kmeans.fit_predict(data)

print("Number of data points in each cluster:", Counter(labels))

print('time to get Kmeans fit:', time.time()-start)


score = silhouette_score(data, labels, metric='euclidean',  sample_size=None)

print('time to silhouette_score:', time.time()-start)

unencoded_data = np.load('unencoded_call_data.npy')
unencoded_data = unencoded_data[:10000]

#for line, lab in zip(unencoded_data, labels):
	#print(lab, line)



print("Score", score)

