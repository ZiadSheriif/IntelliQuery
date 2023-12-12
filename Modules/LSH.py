import numpy as np
import os
import struct
import math
import time
from scipy.spatial.distance import cosine
from utils import *

# from best_case_implementation import VecDBBest


# TODO:
# * 1) Build LSH function (indexing)
# * 2) Build semantic query function (retrieval)


def LSH_index(file_path, nbits, index_path, chunk_size=1000, d=70):
    # Calculate the size and number of records
    file_size = os.path.getsize(file_path)
    # print("File size:", file_size)
    record_size = struct.calcsize(f"I{d}f")
    n_records = file_size // record_size
    no_chunks = math.ceil(n_records / chunk_size)
    # print("Number of records:", n_records)
    # print("Number of chunks:", no_chunks)

    plane_norms = np.random.rand(nbits, d) - 0.5
    os.makedirs(index_path, exist_ok=True)

    for i in range(no_chunks):
        data_chunk = read_binary_file_chunk(file_path, f"I{d}f", start_index=i * chunk_size, chunk_size=chunk_size)

        if data_chunk:
            # Vectorized dot product for chunk data
            chunk_vectors = np.array([entry['embed'] for entry in data_chunk])
            data_dot_products = np.dot(chunk_vectors, plane_norms.T)

            # Vectorized decision making
            data_set_decision_hamming = (data_dot_products > 0).astype(int)

            # Writing data to files within the chunk loop
            for j, decision in enumerate(data_set_decision_hamming):
                hash_str = "".join(decision.astype(str))
                file_path = os.path.join(index_path, hash_str + ".txt")
                with open(file_path, "a") as file:
                    file.write(str(data_chunk[j]["id"]) + "\n")

    return plane_norms


def get_top_k_hamming_distances(query, buckets, top_k):
    # Convert buckets to a NumPy array for vectorized operations
    bucket_array = np.array(buckets)

    # Vectorized Hamming distance calculation
    hamming_distances = np.sum(bucket_array != query, axis=1)

    # Getting top K indices
    top_k_indices = np.argsort(hamming_distances)[:top_k]
    top_k_distances = hamming_distances[top_k_indices]
    top_k_buckets = bucket_array[top_k_indices]

    return list(zip(top_k_buckets, top_k_distances))


def read_text_files_in_folder(folder_path):
    text_files_content = {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a text file
        if filename.endswith(".txt") and os.path.isfile(file_path):
            # Read the content of the text file
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                # Store content in the dictionary with the filename as the key
                text_files_content[filename] = content

    return text_files_content


def semantic_query_lsh(query, plane_norms, index_path, top_k_neighbours=6):
    """
    Function to Query the LSH indexing
    query: query vector
    plane_norms: random hyperplanes used for partitioning
    index_path: path of the Index to be searched in
    top_k_neighbours: number of neighbouring buckets to consider
    """
    # Compute the hash for the query
    query_dot = np.dot(query, plane_norms.T)
    query_decision = (query_dot > 0).astype(int).squeeze()
    if query_decision.ndim == 0:
        query_decision = np.array([query_decision])
    query_hash_str = "".join(map(str, query_decision))

    # Read all bucket filenames in the index folder
    bucket_filenames = [f for f in os.listdir(index_path) if f.endswith('.txt')]

    # Convert filenames to binary arrays for Hamming distance calculation
    bucket_binary_arrays = [np.array(list(map(int, filename[:-4]))) for filename in bucket_filenames]

    # Calculate Hamming distances for all buckets
    hamming_distances = np.array([np.sum(bucket != query_decision) for bucket in bucket_binary_arrays])

    # Get the indices of the buckets with the smallest Hamming distances
    nearest_bucket_indices = np.argsort(hamming_distances)[:top_k_neighbours]

    # Aggregate results from nearest buckets
    index_result = []
    for idx in nearest_bucket_indices:
        bucket_filename = bucket_filenames[idx]
        file_path = os.path.join(index_path, bucket_filename)
        try:
            list_ids = np.loadtxt(file_path, dtype=int)
            index_result.extend(np.atleast_1d(list_ids).tolist())
        except FileNotFoundError:
            print(f"The file {file_path} doesn't exist. Skipping.")

    return query_hash_str, np.array(index_result)

