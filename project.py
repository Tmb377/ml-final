import csv
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


#data columns to use
#data_cols = ["Unique Key", "Created Date", "Agency", "Agency Name", "Complaint Type", "Descriptor", "Location Type",  "City", "Status", "Resolution Description", "Resolution Action Updated Date",  "Latitude", "Longitude"]
data_cols = ["Unique Key",  "Agency", "Complaint Type",  "Latitude", "Longitude"]

#data types
data_types = {'Unique Key':np.int64,  'latitude':np.float64, 'longitude':np.float64 }

#name of csv file
file_name = '311_Service_Requests.csv'

#pandas csv parser
data = pd.read_csv(file_name, usecols=data_cols, dtype=data_types, parse_dates=True, engine='c',infer_datetime_format=True, float_precision='round-trip',keep_default_na=False)

categorical_feats = [ "Agency", "Complaint Type"]


#another way of parsing csv files, reads as dictioary instead of vectors
#dict_vect = DictVectorizer()
#dict_vect.fit_transform(data).toArray()

#encodes categorical features
#enc = OneHotEncoder( categorical_features = categorical_feats, handle_unknown='ignore')
#num_data = enc.fit_transform(data)

#fills missing features
#imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
#filled_data = imp.fit_transform(data[""])
#imp.fit(data)

