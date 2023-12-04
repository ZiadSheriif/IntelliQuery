import numpy as np
import os


from scipy.spatial.distance import cosine


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

    # buckets = {}
    # {id:1:[1,58,,2]}

    for item in data:
        vector = item["embed"]
        id = item["id"]

        # Dot Product with Random Planes
        data_dot_product = np.dot(vector, plane_norms.T)

        # Decision Making
        data_set_decision_hamming = (data_dot_product > 0) * 1

        # Bucket no. (Key)
        hash_str = "".join(data_set_decision_hamming.astype(str))  # 101001101

        # TODO Write as batches
        # Add This vector to the bucket
        file_path = os.path.join(index_path, hash_str + ".txt")

        # Open File in Append Mode
        with open(file_path, "a") as file:
            file.write(str(id) + "\n")

        # if hash_str not in buckets.keys():
        #     buckets[hash_str].append(id)
        #     buckets[hash_str] = []

    return plane_norms
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
    # Bucket no. (Key)
    hash_str = "".join(query_dot.astype(str))  # 101001101

    file_path = os.path.join(index_path, hash_str + ".txt")
    result = read_text_files_in_folder(index_path)
    
    list_buckets = []
    for filename, content in result.items():
        list_buckets.append(list(map(int, filename[:-4])))
    min_ham_buckets = get_top_k_hamming_distances(query_dot, list_buckets, 3)
    # print("query_dot",query_dot)
    # print("min_ham_buckets",min_ham_buckets)

   
    # TODO @Ziad Sherif
    # if query_hash_str in buckets.keys():
    #     bucket_containing_query = buckets[query_hash_str]
    #     min_dist=100
    #     index=-1
    #     for vec in bucket_containing_query:
    #         print(dataset[vec])
    #         print(query)
    #         dot_res=np.dot(query,dataset[vec].T)
    #         print(dot_res)
    #         res = dot_res / (np.linalg.norm(query) * np.linalg.norm(dataset[vec].T))
    #         if res<min_dist:
    #             min_dist=res
    #             index=vec

    #     print(index)
    #     print("Query belongs to bucket:", bucket_containing_query)
    # else:
    #     print("Query doesn't match any existing buckets.")
    # return query_dot

    # Go to that file and just simply return buckets in it :D

    file_path = os.path.join(index_path, hash_str + ".txt")
    index_result =[]
    for (bucket, hamming_distance) in min_ham_buckets:
        file_path = os.path.join(index_path, "".join(map(str,bucket)) + ".txt")
        try:
            # index_result = np.loadtxt(file_path, dtype=int)
            list_1 = np.loadtxt(file_path, dtype=int, ndmin=1)
            # print("list_1",list_1.shape)
            
            index_result += list_1.tolist()
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print(
                f"The file {file_path} doesn't exist. Setting index_result to a default value."
            )
            index_result = []
        # index_result = np.loadtxt(os.path.join(index_path, hash_str+'.txt'),dtype=int)
        # if len(index_result) == 0:
        #     print("Query doesn't match any existing buckets.")
        #     return hash_str, np.array([99999999999999999])
    return hash_str, np.array(index_result)  # Bucket no
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

    # print("target_vector",target_vector)
    # print("data",data)

    # Calculate cosine similarities using vectorized operations
    # print("target_vector",target_vector)
    # print("data",data[1:5])
    
    
    # Solving the problem when we need elements more than the elements in bucket
    k = min(data.shape[0] - 1,k)

    
    similarities = 1 - np.array([cosine(target_vector.T.squeeze(), vector) for vector in data])

    # Find the indices of the top k most similar vectors
    most_similar_indices = np.argpartition(-similarities, k)[:k]

    # Retrieve the top k most similar vectors
    print("most_similar_indices",most_similar_indices)
    k_most_similar_vectors = data[most_similar_indices]

    return most_similar_indices, k_most_similar_vectors
