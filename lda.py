import csv, io, pickle, time, random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim import models


def preprocessing():
	data_cols = ['Incident Zip','Complaint Type', 'Created Date']
	data = pd.read_csv('all_Service_Requests.csv', usecols=data_cols,   engine='c', float_precision='round-trip')
	data = data.dropna(axis = 0, how='any')

	#filter out non-nyc zip codes
	zip_code_file = open('all_nyc_zips.txt')
	all_zips = sorted(zip_code_file.read().split())
	zip_filter = (data['Incident Zip'].isin(all_zips))
	data = data.loc[zip_filter]
	all_zips = [int(x) for x in all_zips]
	data['Incident Zip'] = data['Incident Zip'].astype(int)
	#only need month
	data['Created Date'] = data['Created Date'].apply(lambda x: int(str(x)[:2]))

	#list of complaint types
	complaint_list = np.sort(data['Complaint Type'].unique())
	#list of dicts: index is zip code, dict keys are complaint types
	documents = [dict.fromkeys(complaint_list, 0) for x in range(len(all_zips))]
	for line in data.itertuples():
		documents[all_zips.index(line[3])][line[2]] += 1
	#get rid of dictionary keys, turn into list of complaint type counts
	processed_data = [list(x.values()) for x in documents]
	np.save('ziplist.npy', all_zips)
	np.save('complist.npy', complaint_list)
	np.save('ldadata.npy', processed_data)

def get_saved_data():
	return (np.load('ldadata.npy'), np.load('ziplist.npy'), np.load('complist.npy'))

def lda_model(data, topics):
	lda_algo = LDA(n_topics=topics, random_state = 10)
	new_data = lda_algo.fit_transform(data)
	print("Data complexity:", lda_algo.perplexity(data))
	return new_data

preprocessing()

data, zips, comps = get_saved_data()

processed_data = lda_model(data, len(data[0]))

for z, line in zip(zips, processed_data):
	most_common = np.argmax(line)
	print(z, comps[np.argmax(line)])



