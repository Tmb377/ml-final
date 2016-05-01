import csv, io, pickle, time, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


start = time.time()
def get_data():

    data = np.load('call_data.npy')

    print('time to get data:', time.time()-start)

    data = data[:10000]
    return(data)

def get_kmeans(data,n):
    #default is 8 clusters
    kmeans = KMeans(n_clusters=n, max_iter=20)
    labels = kmeans.fit_predict(data)

    print('time to get Kmeans fit:', time.time()-start)
    return(kmeans)


def get_silhouette(data, labels):
    score = silhouette_score(data, labels, metric='euclidean',  sample_size=None)
    print('time to silhouette_score:', time.time()-start)
    return(score)
