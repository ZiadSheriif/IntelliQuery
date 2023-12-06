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

    list_buckets = []
    for filename, content in result.items():
        list_buckets.append(list(map(int, filename[:-4])))
    number_of_neighbours = 6
    min_hamming_buckets = get_top_k_hamming_distances(query_dot, list_buckets, number_of_neighbours)
    index_result =[]
    for (bucket, hamming_distance) in min_hamming_buckets:
        file_path = os.path.join(index_path, "".join(map(str,bucket)) + ".txt")
        try:
            list_1 = np.loadtxt(file_path, dtype=int)
            list_buckets = np.atleast_1d(list_1).tolist()
            index_result+=list_buckets

        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print(f"The file {file_path} doesn't exist. Setting index_result to a default value.")
            index_result = []
    return hash_str, np.array(index_result) # Bucket no
    # return index_result



