import numpy as np
import os


# TODO:
#* 1) Build LSH function (indexing)
#* 2) Build semantic query function (retrieval)

def LSH_index(data, nbits,index_path, d=70):
    '''
    Function to Build the LSH indexing
    data:[{'id':int,'embed':vector}]
    nbits: no of bits of the Buckets
    index_path:path of the Result to be saved
    d: vector dimension
    '''
    # create nbits Random hyperplanes used for portioning

    plane_norms = np.random.rand(nbits, d) - 0.5



    # If index Folder Doesn't Exist just Create it :D
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    # buckets = {}
    # {id:1:[1,58,,2]}

    for item in data:
        vector=item['embed']
        id=item['id']

        # Dot Product with Random Planes
        data_dot_product = np.dot(vector, plane_norms.T)
      
        # Decision Making
        data_set_decision_hamming = (data_dot_product > 0)*1

        # Bucket no. (Key)
        hash_str = ''.join(data_set_decision_hamming.astype(str)) # 101001101

        # TODO Write as batches
        # Add This vector to the bucket
        file_path = os.path.join(index_path, hash_str+'.txt')

        # Open File in Append Mode
        with open(file_path, "a") as file:
            file.write(str(id) + "\n")

        # if hash_str not in buckets.keys():
        #     buckets[hash_str].append(id)
        #     buckets[hash_str] = []
            
    return plane_norms
    

def semantic_query_lsh(query, plane_norms,index_path):
    '''
    Function to Query the LSH indexing
    query:[] query vector
    plane_norms: [[]]
    index_path:path of the Index to be Search in
    '''
    # Dot Product with Random Planes
    query_dot = np.dot(query, plane_norms.T)

    # Decision Making
    query_dot = (query_dot > 0)*1

    # Bucket no. (Key)
    hash_str = ''.join(query_dot.astype(str)) # 101001101

    #TODO @Ziad Sherif
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

    file_path = os.path.join(index_path, hash_str+'.txt')

    try:
        index_result = np.loadtxt(file_path, dtype=int)
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        print(f"The file {file_path} doesn't exist. Setting index_result to a default value.")
        index_result = []  
    # index_result = np.loadtxt(os.path.join(index_path, hash_str+'.txt'),dtype=int)
    
    return hash_str,index_result #Bucket no
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

