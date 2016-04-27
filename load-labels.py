import cPickle as pickle
# import project
from sklearn.cluster import KMeans

labels = pickle.load(open('labels.p','rb'))

kmeans = KMeans()
print kmeans.get_params(labels.all())
