import numpy as np
from sklearn.cluster import KMeans


class InvertedFileSystem:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.inverted_files = [[] for _ in range(n_clusters)]
        self.centroids = None

    def build_index(self, data):
        # Cluster the data
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_

        # Build inverted files
        for idx, label in enumerate(labels):
            self.inverted_files[label].append(idx)

    def query(self, vector, top_k=5):
        # Assign vector to nearest cluster
        nearest_cluster = np.argmin(np.linalg.norm(self.centroids - vector, axis=1))

        # Search in the corresponding inverted file
        candidates = self.inverted_files[nearest_cluster]
        distances = [np.linalg.norm(vector - data[idx]) for idx in candidates]
        nearest_indices = np.argsort(distances)[:top_k]

        return [candidates[i] for i in nearest_indices]


def brute_force_search(query_vector, data, top_k=5):
    # Calculate distances from the query vector to all vectors in the dataset
    distances = np.linalg.norm(data - query_vector, axis=1)

    # Get the indices of the top k nearest neighbors
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices


# !testing IVFFF
data = np.random.rand(100000, 70)
ivf = InvertedFileSystem(n_clusters=3)
ivf.build_index(data)

query_vector = np.random.rand(70)

# Perform brute force search
brute_force_results = brute_force_search(query_vector, data, top_k=10)
print("Brute force top k: ", brute_force_results)
top_k_results = ivf.query(query_vector, top_k=10)
print("top k: ", top_k_results)


# Get intersection
brute_force_set = set(brute_force_results)
ivf_set = set(top_k_results)

intersection = brute_force_set.intersection(ivf_set)
print("Intersection of Brute Force and IVF: ", intersection)
print("length of the intersection: ", len(intersection))
