import csv, io, pickle, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

start = time.time()
#data columns to use
#data_cols = ["Unique Key", "Created Date", "Agency", "Agency Name", "Complaint Type", "Descriptor", "Location Type",  "City", "Status", "Resolution Description", "Resolution Action Updated Date",  "Latitude", "Longitude"]
data_cols = ["Unique Key",  "Agency", "Complaint Type",  "Latitude", "Longitude", "Status"]

#data types
data_types = {'Unique Key':np.int64,  'latitude':np.float64, 'longitude':np.float64 }

#name of csv file
file_name = '311_Service_Requests.csv'

#pandas csv parser
data = pd.read_csv(file_name, usecols=data_cols, dtype=data_types, parse_dates=True, engine='c',infer_datetime_format=True, float_precision='round-trip')

print('time to get from csv:', time.time()-start)

#drop any row with missing values
processed_data = data.dropna(subset=data_cols, how='any')

print('time to get processed data:', time.time()-start)
processed_data = processed_data[:10000]

#get_dummies replaces categorical features with binary features
encoded_data = pd.get_dummies(processed_data)
#save data into a pickle
# pickle.dump(encoded_data, open('encoded_data.p','wb'))

print('time to get encoded_data:', time.time()-start)

#default is 8 clusters
kmeans = KMeans()
labels = kmeans.fit_predict(encoded_data)

print('time to get Kmeans fit:', time.time()-start)


score = silhouette_score(encoded_data, labels, metric='euclidean')

#
print('time to silhouette_score:', time.time()-start)
>>>>>>> c4ee9503cf32449c498394fc2edd7580993f4712

#print(labels)
print("Score", score)


#save labels into a pickle

# pickle.dump(labels, open('labels.p','wb'))
