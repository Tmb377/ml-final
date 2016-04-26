import csv
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


#data columns to use
#data_cols = ["Unique Key", "Created Date", "Agency", "Agency Name", "Complaint Type", "Descriptor", "Location Type",  "City", "Status", "Resolution Description", "Resolution Action Updated Date",  "Latitude", "Longitude"]
data_cols = ["Unique Key",  "Agency", "Complaint Type",  "Latitude", "Longitude", "Status"]

#data types
data_types = {'Unique Key':np.int64,  'latitude':np.float64, 'longitude':np.float64 }

#name of csv file
file_name = '311_Service_Requests.csv'

#pandas csv parser
data = pd.read_csv(file_name, usecols=data_cols, dtype=data_types, parse_dates=True, engine='c',infer_datetime_format=True, float_precision='round-trip')

#drop any row with missing values
processed_data = data.dropna(subset=data_cols, how='any')


#get_dummies replaces categorical features with binary features
encoded_data = pd.get_dummies(processed_data)



kmeans = KMeans()
labels = kmeans.fit_predict(encoded_data)



