import kmeans as km
import pylab
from sklearn.decomposition import PCA


data = km.get_data()

#reduce dimensions
pca = PCA(n_components=2)
data = pca.fit_transform(data)
for line in data:
	print(line[0])
labels, kmeans = km.get_kmeans(data,10)
clusters = kmeans.cluster_centers_

columns = ['Unique Key', 'Complaint Type', 'Incident Zip', 'Latitude', 'Longitude', 'month',
       'year', 'hour', 'AM', 'Agency_3-1-1', 'Agency_CHALL',
       'Agency_DCA', 'Agency_DEP', 'Agency_DFTA', 'Agency_DHS',
       'Agency_DOB', 'Agency_DOE', 'Agency_DOF', 'Agency_DOHMH',
       'Agency_DOITT', 'Agency_DOT', 'Agency_DPR', 'Agency_DSNY',
       'Agency_EDC', 'Agency_FDNY', 'Agency_HPD', 'Agency_HRA',
       'Agency_NYPD', 'Agency_TLC', 'Status_Assigned', 'Status_Closed',
       'Status_Email Sent', 'Status_Open', 'Status_Pending',
       'Status_Started']

#x,y = 4, 3

colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1], [.1, 1, 0], [.5, .6, .2], [.2, .6, .5], [.6, .2, .5], [0, .9, .0], [.9, 0, 0], [0, 0, .9])[i] for i in labels])
pylab.scatter(data[:,0],data[:,1],color=colors)
pylab.scatter(clusters[:,0],clusters[:,1], marker='x', s = 500, linewidths=2)
pylab.xlabel(columns[0])
pylab.ylabel(columns[1])
#pylab.ylim([9750,12000])
#pylab.xlim([-10, 160])
pylab.show()

