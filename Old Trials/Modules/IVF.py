import os
import numpy as np
from sklearn.cluster import KMeans
import time
from scipy.spatial.distance import cosine


class InvertedFileSystem:
    def __init__(self, n_clusters, data_dir):
        self.n_clusters = n_clusters
        self.data_dir = data_dir
        self.inverted_file_paths = [
            os.path.join(data_dir, f"inverted_file_{i}.npy") for i in range(n_clusters)
        ]
        self.centroids = None

    def build_index(self, data):
        # Cluster the data
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(data)
        self.centroids = kmeans.cluster_centers_

        # Build inverted files
        inverted_files = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(labels):
            inverted_files[label].append(idx)

        # Save inverted files to disk
        for i, inverted_file in enumerate(inverted_files):
            np.save(self.inverted_file_paths[i], inverted_file)

    def query(self, vector, top_k=5):
        # Assign vector to nearest cluster
        nearest_cluster = np.argmin(np.linalg.norm(self.centroids - vector, axis=1))

        # Load the corresponding inverted file from disk
        inverted_file = np.load(self.inverted_file_paths[nearest_cluster])

        # Search in the inverted file
        distances = [np.linalg.norm(vector - data[idx]) for idx in inverted_file]
        nearest_indices = np.argsort(distances)[:top_k]

        return [inverted_file[i] for i in nearest_indices]


def brute_force_cosine_similarity(query_vector, data, top_k=5):
    # Calculate cosine similarities for each vector in the dataset
    similarities = [1 - cosine(query_vector, vector) for vector in data]

    # Get the indices of the top k most similar vectors
    nearest_indices = np.argsort(similarities)[-top_k:]

    # Return the indices and their cosine similarities
    return [idx for idx in reversed(nearest_indices)]

def run_queries(n_queries, ivf, data, top_k=5):
    total_time_ivf = 0
    total_time_brute_force = 0
    total_score_ivf = 0
    ivf_results = []
    brute_force_results = []

    for _ in range(n_queries):
        query_vector = np.random.rand(70)

        start_time = time.time()
        ivf_result = ivf.query(query_vector, top_k)
        end_time = time.time()
        total_time_ivf += end_time - start_time
        ivf_results.append(ivf_result)

        start_time = time.time()
        brute_force_result = brute_force_cosine_similarity(query_vector, data, top_k)
        end_time = time.time()
        total_time_brute_force += end_time - start_time
        brute_force_results.append(brute_force_result)

        intersection = len(set(ivf_result).intersection(brute_force_result))
        total_score_ivf += intersection / top_k  

    avg_time_ivf = total_time_ivf / n_queries
    avg_score_ivf = total_score_ivf / n_queries
    avg_time_brute_force = total_time_brute_force / n_queries

    print(f"IVF: Average time = {avg_time_ivf}, Average score = {avg_score_ivf}")
    print(f"Brute Force: Average time = {avg_time_brute_force}")

    # Calculate intersection of top k results
    intersection = set(ivf_result).intersection(brute_force_result)
    print(f"Intersection of top {top_k} results: {intersection}")

# !testing IVF
data_dir = "inverted_files"
os.makedirs(data_dir, exist_ok=True)
number_of_queries=10
data_set=10000

data = np.random.rand(data_set, 70)
ivf = InvertedFileSystem(n_clusters=5, data_dir=data_dir)
ivf.build_index(data)

print("Dataset in k: ",data_set//1000)
print("Number of Queries: ",number_of_queries)

run_queries(number_of_queries, ivf, data)


# # !testing IVF
# data_dir = "inverted_files"
# os.makedirs(data_dir, exist_ok=True)

# data = np.random.rand(100000, 70)
# ivf = InvertedFileSystem(n_clusters=3, data_dir=data_dir)
# ivf.build_index(data)

# query_vector = np.random.rand(70)


# # brute force search
# start_time = time.time()
# brute_force_results = brute_force_cosine_similarity(query_vector, data, top_k=10)
# brute_force_time = time.time() - start_time
# print("Brute force top k: ", brute_force_results)
# print("Brute force time: ", brute_force_time)
# print("============================================")
# # Timing IVF query
# start_time = time.time()
# top_k_results = ivf.query(query_vector, top_k=10)
# ivf_time = time.time() - start_time
# print("IVF top k: ", top_k_results)
# print("IVF time: ", ivf_time)


# # Get intersection
# brute_force_set = set(brute_force_results)
# ivf_set = set(top_k_results)

# intersection = brute_force_set.intersection(ivf_set)
# print("Intersection of Brute Force and IVF: ", intersection)
# print("length of the intersection: ", len(intersection))

# print("********************************************")
