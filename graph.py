import kmeans as km
import pylab


data = km.get_data()
kmeans = km.get_kmeans(data,10)
clusters = kmeans.cluster_centers_


x,y = 1,2

pylab.scatter(data[:,x],data[:,y],color='green')
pylab.scatter(clusters[:,x],clusters[:,y], marker='x', s = 500, linewidths=2)

pylab.show()
