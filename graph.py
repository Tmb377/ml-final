import kmeans as km
import pylab


data = km.get_data()
kmeans = km.get_kmeans(data,2)
clusters = kmeans.cluster_centers_

<<<<<<< HEAD
x,y = 5,104
=======

x,y = 27,28
>>>>>>> ce7f19d28bcebeb2c1341395d190b0b680895d92

pylab.scatter(data[:,x],data[:,y],color='green')
pylab.scatter(clusters[:,x],clusters[:,y], marker='x', s = 500, linewidths=2)

pylab.show()
