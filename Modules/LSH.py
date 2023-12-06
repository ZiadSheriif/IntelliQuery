import numpy as np
import os


from scipy.spatial.distance import cosine
# from best_case_implementation import VecDBBest


# TODO:
# * 1) Build LSH function (indexing)
# * 2) Build semantic query function (retrieval)


def LSH_index(data, nbits, index_path, d=70):
    """
    Function to Build the LSH indexing
    data:[{'id':int,'embed':vector}]
    nbits: no of bits of the Buckets
    index_path:path of the Result to be saved
    d: vector dimension
    """
    # create nbits Random hyperplanes used for portioning

    plane_norms = np.random.rand(nbits, d) - 0.5

    # If index Folder Doesn't Exist just Create it :D
    if not os.path.exists(index_path):
        os.makedirs(index_path)


    for item in data:
        vector = item["embed"]
        id = item["id"]

        # Dot Product with Random Planes
        data_dot_product = np.dot(vector, plane_norms.T)

        # Decision Making
        data_set_decision_hamming = (data_dot_product > 0) * 1

        # Bucket no. (Key)
        hash_str = "".join(data_set_decision_hamming.astype(str))  # 101001101

        # Add This vector to the bucket
        file_path = os.path.join(index_path, hash_str + ".txt")

        # Open File in Append Mode
        with open(file_path, "a") as file:
            file.write(str(id) + "\n")

        # if hash_str not in buckets.keys():
        #     buckets[hash_str].append(id)
        #     buckets[hash_str] = []

    return plane_norms

def get_top_k_hamming_distances(query, buckets, top_k):
    distances = []
    # Calculate Hamming distance for each bucket
    for bucket in buckets:
        hamming_distance = sum(bit1 != bit2 for bit1, bit2 in zip(query, bucket))
        distances.append((bucket, hamming_distance))
    # Sort distances and get the top K
    sorted_distances = sorted(distances, key=lambda x: x[1])
    top_k_distances = sorted_distances[:top_k]
    return top_k_distances
def read_text_files_in_folder(folder_path):
    text_files_content = {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a text file
        if filename.endswith('.txt') and os.path.isfile(file_path):
            # Read the content of the text file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Store content in the dictionary with the filename as the key
                text_files_content[filename] = content

    return text_files_content

# # Replace 'your_folder_path' with the actual path of the folder containing text files
# folder_path = 'your_folder_path'
# result = read_text_files_in_folder(folder_path)

# # Print or manipulate the result as needed
# for filename, content in result.items():
#     print(f"File: {filename}\nContent:\n{content}\n{'=' * 30}\n")


def semantic_query_lsh(query, plane_norms, index_path):
    
    
    """
    Function to Query the LSH indexing
    query:[] query vector
    plane_norms: [[]]
    index_path:path of the Index to be Search in
    """
    # Dot Product with Random Planes
    query_dot = np.dot(query, plane_norms.T)

    # Decision Making
    query_dot = (query_dot > 0) * 1

    query_dot = query_dot.squeeze()
     # Ensure query_dot is 1D for string conversion
    if query_dot.ndim == 0:
        query_dot = np.array([query_dot])
    # Bucket no. (Key)
    # hash_str = "".join(query_dot.astype(str))  # 101001101
    hash_str = "".join(map(str, query_dot.astype(int)))  # Converts boolean array to int and then to string

    file_path = os.path.join(index_path, hash_str + ".txt")
    result = read_text_files_in_folder(index_path)

# # Print or manipulate the result as needed
    list_buckets = []
    for filename, content in result.items():
        list_buckets.append(list(map(int, filename[:-4])))
    min_ham_buckets = get_top_k_hamming_distances(query_dot, list_buckets, 5)
    # print("query_dot",query_dot)
    # print("min_ham_buckets",min_ham_buckets)
    index_result =[]
    for (bucket, hamming_distance) in min_ham_buckets:
        file_path = os.path.join(index_path, "".join(map(str,bucket)) + ".txt")
        try:
            list_1 = np.loadtxt(file_path, dtype=int)
            list_buckets = np.atleast_1d(list_1).tolist()
            index_result+=list_buckets

        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print(
                f"The file {file_path} doesn't exist. Setting index_result to a default value."
            )
            index_result = []
    return hash_str, np.array(index_result) # Bucket no
    # return index_result


# # Write data to a text file
# file_path = "../random_data.txt"
# read_data = np.loadtxt(file_path)
# plane_norms = LSH(read_data, 8)
# # query = np.random.random((1, 70))
# # print(query)
# query=[read_data[0]]
# print(query)
# folder_name = "bucket_files"
#
# print(result)


def get_top_k_similar(target_vector, data, k=5):
    """
    Find the top k most similar vectors in data to a given target_vector.

    :param target_vector: The vector to compare against.
    :param data: Dataset of vectors.
    :param k: The number of most similar vectors to find.
    :return: Indices of the top k most similar vectors, and the vectors themselves.
    """
    if len(data) < k:
        print("Error: k is larger than the size of the dataset.")
        k = len(data)

    # print("target_vector",target_vector)
    # print("data",data)

    # Calculate cosine similarities using vectorized operations
    # print("target_vector",target_vector)
    # print("data",data[1:5])
    # print(data.shape)
    
    similarities = 1 - np.array([cosine(target_vector.T.squeeze(), vector) for vector in data])

    # Find the indices of the top k most similar vectors
    most_similar_indices = np.argpartition(-similarities, k)[:k]

    # Retrieve the top k most similar vectors
    # print("most_similar_indices",most_similar_indices)
    k_most_similar_vectors = data[most_similar_indices]

    return most_similar_indices, k_most_similar_vectors



def top_k_cosine_similarity(query,data, k=5):
    """
    Find the top k elements in the data with the highest cosine similarity to the query.

    :param data: A 2D array where each row is a vector.
    :param query: A 1D array representing the query vector.
    :param k: The number of top elements to return.
    :return: Indices and values of the top k elements with highest cosine similarity.
    """
    # Check if k is larger than the dataset
    if len(data) < k:
        print("Error: k is larger than the size of the dataset.")
        k = len(data)


    # Compute cosine similarity
    cosine_similarities = np.array([1 - cosine(query, vector) for vector in data])

    # Get the indices of the top k elements
    top_k_indices = np.argsort(cosine_similarities)[-k:]

    # Get the top k elements
    top_k_values = data[top_k_indices]

    return top_k_indices, top_k_values

