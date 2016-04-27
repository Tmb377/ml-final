import pickle
import matplotlib as plt
from sklearn.cluster import KMeans
import numpy as np

#load data from pickles
labels = pickle.load(open('labels.p','rb'))
# encoded_data = pickle.load(open('encoded_data.p','rb'))

kmeans = KMeans()
print(kmeans.get_params(labels.all()))
