import sys, csv
import numpy as np
import lda


def groups():
	data, zips, comps = lda.get_saved_data()

	file_names = ['two_topics.csv', 'three_topics.csv', 'four_topics.csv', 'five_topics.csv']
	info_file_names = ['two_topic_info.txt', 'three_topic_info.txt', 'four_topic_info.txt', 'five_topic_info.txt']

	for file_name, info_file, topic_num in zip(file_names, info_file_names, range(2, 6)):
			processed_data, components = lda.lda_model(data, topic_num)
			all_data = []
			topic_dist_count = np.zeros(topic_num)
			for i in range(len(processed_data)):
				topic_dist_count[np.argmax(processed_data[i])] += 1
				new_line = {'zip':zips[i//12], 'month':i%12+1, 'topic':np.argmax(processed_data[i])}
				all_data.append(new_line)
			with open(file_name, 'w') as f:
				writer = csv.DictWriter(f, fieldnames=['zip', 'month', 'topic'])
				writer.writerows(all_data)
			with open(info_file, 'w') as f:
				f.write("Number of Topics %d\n" % topic_num)
				f.write("Documents per Topic %s\n" % str(topic_dist_count))
				f.write("\n")
				for i in range(len(components)):
					f.write("\n")
					f.write("Topic number %d\n" % (i+1))
					sorted_list = np.array(components[i]).argsort()[::-1]
					for complaint in sorted_list[:10]:
						f.write("%s : %f\n" % (comps[complaint], components[i][complaint]))


if __name__ == "__main__":
	groups()