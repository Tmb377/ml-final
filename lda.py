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
	#list of dicts: index is zip code each month, dict keys are complaint types
	documents = [dict.fromkeys(complaint_list, 0) for x in range(len(all_zips) * 12)]
	for line in data.itertuples():
		index = all_zips.index(line[3]) * 12 + line[1] - 1
		documents[index][line[2]] += 1
	#get rid of dictionary keys, turn into list of complaint type counts
	processed_data = [list(x.values()) for x in documents]
	np.save('ziplist.npy', all_zips)
	np.save('complist.npy', complaint_list)
	np.save('ldadata.npy', processed_data)

def get_saved_data():
	return (np.load('ldadata.npy'), np.load('ziplist.npy'), np.load('complist.npy'))

def lda_model(data, topics):	
	lda_algo = LDA(n_topics=topics, random_state = 1)
	new_data = lda_algo.fit_transform(data)
	print("Data complexity:", lda_algo.perplexity(data))
	return (new_data, lda_algo.components_)

#preprocessing()

#data, zips, comps = get_saved_data()

#with open('topic_distribution.txt', 'w') as f:
#	for i in range(2, 20):
#		f.write("Number of topics: %d" %  i)
#		processed_data, components = lda_model(data, 10)
#		for topic in components:
#			sorted_comps = np.argsort(topic)
#			for x in range(10):
#				f.write("%d: %s %f\n"% (x+1, comps[sorted_comps[x]], topic[sorted_comps[x]]))
#		f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#for topic in components:
#	sorted_topic = np.argsort(topic)
#	print("Distribution")
#	for comp in sorted_topic[:10]:
#		print("Topic:", comps[comp], "Dist:", topic[comp])

#print(processed_data.components)
#for i in range(2, 10):
#	processed_data = lda_model(data, i)
#	for line in processed_data:
#		print("Top five of topic", i)
#		sorted_line = np.argsort(line)
#		for item in sorted_line[:5]:
#			print(comps[item], processed_data[item])

#top_complaint = [comps[np.argmax(x)] for x in processed_data]


#top_three_comp = [[sorted(range(len(processed_data)), key=lambda i: processed_data[i])[-3:]] for x in processed_data]
#new_list = [[x, comps[np.argmax(y)], 'USA'] for x, y in zip(zips, processed_data)]
#new_list = []
#for z, line in zip(zips, processed_data):
#	for i in range(1, 13):
#		new_line = [i, z, comps[np.argmax(line)], 'USA']
#		new_list.append(new_line)

#for i in range(len(top_complaint)):
#	new_line = [i%12 + 1, zips[i//12], top_complaint[i]]
#	new_list.append(new_line)

#for i in range(len(top_three_comp)):
#	new_line = [zips[i//12], i%12+1, top_three_comp[i][0], top_three_comp[i][1], top_three_comp[i][2]]
#	new_list.append(new_line)


#with open("top_tree.csv", 'w') as f:
#	writer = csv.writer(f, fieldnames = ['Zip', 'Month', 'Complaint 1', 'Complaint 2', 'Complaint 3'])
#	writer.writerows(new_list)


#with open("months_csv.csv", 'w') as f:
#	writer = csv.writer(f)
#	writer.writerows(new_list)
#for z, line in zip(zips, processed_data):
	
#	three_most_common = np.argmax(line)
	
#	print(z, comps[three_most_common])
	



