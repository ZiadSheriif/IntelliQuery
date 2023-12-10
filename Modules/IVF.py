import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.spatial.distance import cosine



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


def brute_force_cosine_similarity(query_vector, data, top_k=5):
    # Calculate cosine similarities for each vector in the dataset
    similarities = [1 - cosine(query_vector, vector) for vector in data]

    # Get the indices of the top k most similar vectors
    nearest_indices = np.argsort(similarities)[-top_k:]

    # Return the indices and their cosine similarities
    top_k_cosine_similarities = [idx for idx in reversed(nearest_indices)]
    return top_k_cosine_similarities


# !testing IVFFF
data = np.random.rand(1000, 70)
ivf = InvertedFileSystem(n_clusters=3)
ivf.build_index(data)

query_vector = np.random.rand(70)

# Timing brute force search
start_time = time.time()
brute_force_results = brute_force_cosine_similarity(query_vector, data, top_k=10)
brute_force_time = time.time() - start_time
print("Brute force top k: ", brute_force_results)
print("Brute force time: ", brute_force_time)

print("==========================================")
# Timing IVF query
start_time = time.time()
top_k_results = ivf.query(query_vector, top_k=10)
ivf_time = time.time() - start_time
print("IVF top k: ", top_k_results)
print("IVF time: ", ivf_time)


# Get intersection
brute_force_set = set(brute_force_results)
ivf_set = set(top_k_results)

intersection = brute_force_set.intersection(ivf_set)
print("Intersection of Brute Force and IVF: ", intersection)
print("length of the intersection: ", len(intersection))
