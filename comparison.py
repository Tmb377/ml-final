import kmeans as clustering



data = clustering.get_data()


km_labels = clustering.get_kmeans(data,20)

#aff_labels = clustering.get_affinity(data)

#print("Number of affinity clusters", len(set(aff_labels)))

print("Score for Kmeans:", clustering.get_silhouette(data, km_labels))

#print("Score for Affinity:", clustering.get_silhouette(data, aff_labels))