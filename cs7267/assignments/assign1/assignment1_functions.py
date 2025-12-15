import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin
import random

def simplekmeans(X_df, K, max_runs=300):
    pseudo = 100 * random.randint(1, 100) # generate a pseudo-random seed
    input_df = X_df  #preserve original data frame so the original data can be used for plotting
    centroid_indices = input_df.sample(K, random_state=pseudo, replace=False).index.values # Randomly initialize centroids
    centroids = input_df.values[centroid_indices]

    for x in range(max_runs):
        labels = pairwise_distances_argmin(input_df.values, centroids)

        new_centroids = np.array([input_df.values[labels == i].mean(axis=0) 
                                  for i in range(K)])

        if np.all(new_centroids == centroids):
            break

        centroids = new_centroids

    return centroids, labels
####

find_tss = lambda X: np.sum((X - np.mean(X, axis=0))**2)    #usage: tss = find_tss(X)
find_bcss = lambda tss, wcss: tss - wcss    #usage: bcss = find_bcss(tss, wcss)

def find_wcss(X, K, labels, centers):
    wcss = 0
    m = centers
    for i in range(K):
        x = X[labels == i]
        if x.size > 0:
            wcss += np.sum((x - m[i]) ** 2)
    return wcss
###

def kmtest_plot(X, centers, labels, title):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    plt.scatter(X[0:, 0], X[0:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[0:, 0], centers[0:, 1], c='black', s=200, alpha=0.5, marker='X');
    #plt.show()
    plt.savefig(title)
    plt.close()

def iris_plot(X, centers, labels, title):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    plt.scatter(X[0:, 2], X[0:, 3], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[0:, 2], centers[0:, 3], c='black', s=200, alpha=0.5, marker='X');
    #plt.show()
    plt.savefig(title)
    plt.close()

def cluster_distance_plot(distances, title):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))
    sns.heatmap(distances, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Distances between Original Centers and Best Cluster Centers')
    plt.xlabel('Best Cluster Centers')
    plt.ylabel('Original Centers')
    #plt.show()
    plt.savefig(title)
    plt.close()

def plot_original_result(df, centers, encoder, title):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Use the encoded species to color the data points with a colormap
    encoded_species = encoder.transform(df['species'])

    plt.scatter(df['petal_length'], df['petal_width'], c=encoded_species, s=50, cmap='viridis')
    plt.scatter(centers['petal_length'], centers['petal_width'],c='black', s=200, alpha=0.5, marker='X')
    
    # Hard code the legend to match species names with colors
    color_scheme = {'setosa': 'purple', 'versicolor': 'blue', 'virginica': 'green'}

    legend_elements = []
    for species, color in color_scheme.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=species, markerfacecolor=color, markersize=10))
    plt.legend(handles=legend_elements, title='Species')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    #plt.title(title)
    #plt.show()
    plt.savefig(title)
    plt.close()
