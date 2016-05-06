import csv, io, pickle, time, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation as LDA

def preprocessing():
	data_cols = ['Incident Zip','Complaint Type']
	data_types = {'Incident Zip': np.float64}
	data = pd.read_csv('311_Service_Requests.csv', usecols=data_cols, dtype=data_types, engine='c', float_precision='round-trip', na_values='43123-8895')
	data = data.dropna(axis = 0, how='any')
	#list of zip codes
	zip_list = sorted(data['Incident Zip'].unique())
	#list of complaint types
	complaint_list = np.sort(data['Complaint Type'].unique())
	#list of dicts: index is zip code, dict keys are complaint types
	documents = [dict.fromkeys(complaint_list, 0) for x in range(len(zip_list))]
	for line in data.itertuples():
		documents[zip_list.index(line[2])][line[1]] += 1
	#get rid of dictionary keys, turn into list of complaint type counts
	processed_data = [list(x.values()) for x in documents]
	np.save('ziplist.npy', zip_list)
	np.save('complist.npy', complaint_list)
	np.save('ldadata.npy', processed_data)

def get_saved_data():
	return (np.load('ldadata.npy'), np.load('ziplist.npy'), np.load('complist.npy'))

def lda_model(data, topics):
	lda_algo = LDA(n_topics=topics)
	new_data = lda_algo.fit_transform(data)
	return new_data

preprocessing()

data, zips, comps = get_saved_data()

processed_data = lda_model(data, len(data[0]))

print("Most common complaint for each zip code")
for z, line in zip(zips, processed_data):
	most_common = np.argmax(line)
	print(z, comps[np.argmax(line)])



