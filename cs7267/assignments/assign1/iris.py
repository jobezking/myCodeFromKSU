import pandas as pd
import assignment1_functions as af
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist

iris_full = pd.read_csv('iris.csv', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

iris = iris_full.iloc[:, 0:4].astype(float)  # need separate dataframe containing first 4 attributes

# low wcss is required for good clustering 

center_list = []
label_list = []
wcss_list = []
bcss_list = []
tss_list = []

tss = af.find_tss(iris.values) # calculate total sum of squares
K = 3
for o in range(1,11): # run k-means 10 times to get different results
    centers, labels = af.simplekmeans(iris, K)
    wcss = af.find_wcss(iris.values, K, labels, centers)
    bcss = af.find_bcss(tss, wcss)
    center_list.append(centers)
    label_list.append(labels)
    wcss_list.append(wcss)
    bcss_list.append(bcss)
    tss_list.append(tss)

# find best and worst results based on wcss
best_results = min(wcss_list)
worst_results = max(wcss_list)
best_bcss = af.find_bcss(best_results, tss)
worst_bcss = af.find_bcss(worst_results, tss)

# find index of best and worst results to use to retrieve corresponding centers and labels
best_cluster_index = wcss_list.index(best_results)
worst_cluster_index = wcss_list.index(worst_results)
best_centers = center_list[best_cluster_index]
worst_centers = center_list[worst_cluster_index]
best_labels = label_list[best_cluster_index]
worst_labels = label_list[worst_cluster_index]

# plot best and worst clustering results
af.iris_plot(iris.values, best_centers, best_labels, 'iris-best-clusters.png')
af.iris_plot(iris.values, worst_centers, worst_labels, 'iris-worst-clusters.png')

# Save results of running k-means against iris.csv 10 times to CSV and text files 
df = pd.DataFrame({'TSS': tss_list, 'BCSS': bcss_list, 'WCSS': wcss_list})
df.to_csv('iris-kmeans-results.csv', index=False)

with open('iris_clustering_analysis.txt', 'w') as f:
    f.write(f'Best WCSS: {best_results}, Best Iteration: {best_cluster_index + 1}, \
            Worst WCSS: {worst_results} Worst Iteration: {worst_cluster_index + 1}\n')

#Species data is character. Encode species to integers for analysis
encode_the_species = LabelEncoder()
iris_full['species_encoded'] = encode_the_species.fit_transform(iris_full['species'])

# Normalize the encoded species data
normalize_the_species = MinMaxScaler()
iris_full['species_normalized'] = normalize_the_species.fit_transform(iris_full[['species_encoded']])

#Calculate original centers and values using groupby with respect to attribute 3 and attribute 4
original_centers = iris_full.groupby('species')[['petal_length', 'petal_width']].mean().reset_index()
original_centers_values = original_centers[['petal_length', 'petal_width']].values

#Calculate distances between original centers and best centers
distances = cdist(original_centers_values, best_centers[:, 2:4], metric='euclidean')

# Save distances to text file
with open('distance_original_best_clusters.txt', 'w') as f:
    f.write(f'Distances between original centers and best centers:\n{distances}\n')

# Plot distances between original centers and best centers
af.cluster_distance_plot(distances, 'iris-distance-original-best-clusters.png')

# Plot original species with original centers overlayed
af.plot_original_result(iris_full, original_centers.set_index('species'), encode_the_species, 'iris-original-centers.png')