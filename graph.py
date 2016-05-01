import kmeans as km
import pylab


data = km.get_data()
labels, kmeans = km.get_kmeans(data,10)
clusters = kmeans.cluster_centers_

columns = ['Unique Key', 'Complaint Type',  'Latitude', 'Longitude', 'month',
       'year', 'hour', 'AM', 'Agency_3-1-1', 'Agency_CHALL',
       'Agency_DCA', 'Agency_DEP', 'Agency_DFTA', 'Agency_DHS',
       'Agency_DOB', 'Agency_DOE', 'Agency_DOF', 'Agency_DOHMH',
       'Agency_DOITT', 'Agency_DOT', 'Agency_DPR', 'Agency_DSNY',
       'Agency_EDC', 'Agency_FDNY', 'Agency_HPD', 'Agency_HRA',
       'Agency_NYPD', 'Agency_TLC', 'Status_Assigned', 'Status_Closed',
       'Status_Email Sent', 'Status_Open', 'Status_Pending',
       'Status_Started']

x,y = 1,2

pylab.scatter(data[:,x],data[:,y],color='green')
pylab.scatter(clusters[:,x],clusters[:,y], marker='x', s = 500, linewidths=2)
pylab.xlabel(columns[x])
pylab.ylabel(columns[y])

pylab.show()
