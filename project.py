import csv, io, pickle, time, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

start = time.time()
#data columns to use
data_cols = ["Unique Key",  "Agency", "Complaint Type",  "Latitude", "Longitude", "Status", "Created Date"]

#data types
data_types = {'Unique Key':np.int64,  'latitude':np.float64, 'longitude':np.float64 }

#name of csv file
file_name = '311_Service_Requests.csv'

#pandas csv parser
data = pd.read_csv(file_name, usecols=data_cols, dtype=data_types, parse_dates=True, engine='c',infer_datetime_format=True, float_precision='round-trip')

date_col = data[['Created Date']]


data = data.drop('Created Date', 1)

#process timestamp string
time_data = []
for line in date_col.itertuples():
	info = line[1].split()
	date = info[0].split("/")
	created_time = info[1].split(":")
	am = 0
	if info[2] == 'AM':
		am = 1
	vec = [int(date[0]), int(date[2]), int(created_time[0]), am]
	time_data.append(vec)

time_cols = ['month', 'year', 'hour', 'AM']
time_frame = pd.DataFrame(time_data, columns=time_cols, dtype=np.float64)

data_sets = [data, time_frame]

#combine data with time columns
data = pd.concat(data_sets, axis=1)

print('time to get from csv:', time.time()-start)

#drop any row with missing values
processed_data = data.dropna(axis=0, subset=(time_cols + data_cols[:-1]), how='any')

print('time to get processed data:', time.time()-start)

#np.save('unencoded_call_data', processed_data)

# get unique complaint types
complaint_types = np.sort(processed_data['Complaint Type'].unique())

# replace complaint types with numbers
for i in range(len(complaint_types)):
	processed_data['Complaint Type'] = processed_data['Complaint Type'].replace(complaint_types[i],i)

#get_dummies replaces categorical features with binary features
encoded_data = pd.get_dummies(processed_data)


np.save('call_data', encoded_data)


print('time to get encoded_data:', time.time()-start)



