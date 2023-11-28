import numpy as np
import os

# TODO:
#* 1) Build LSH function (indexing)
#* 2) Build semantic query function (retrieval)

def LSH(data_set, nbits, d=70, folder_name="bucket_files"):
    # create a set of 4 hyperplanes, with 2 dimensions
    plane_norms = np.random.rand(nbits, d) - 0.5
    
    data_set_dot_product = np.dot(data_set, plane_norms.T)
    data_set_decision_hamming = (data_set_dot_product > 0)*1
    
    buckets = {}
    
    for i in range(len(data_set_decision_hamming)):
        hash_str = ''.join(data_set_decision_hamming[i].astype(str)) # 101001101
        if hash_str not in buckets.keys():
            buckets[hash_str] = []
        buckets[hash_str].append(i)
        
        # Create a file for each bucket
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_name = hash_str + ".txt"
        file_path = os.path.join(folder_name, file_name)
        print(file_path)
        with open(file_path, "a") as file:
            file.write(str(i) + "\n")
    
    return plane_norms
    

def semantic_query_lsh(query, plane_norms,folder_name="bucket_files"):
    query_dot = np.dot(query, plane_norms.T)
    query_dot = query_dot > 0
    query_dot = query_dot.astype(int)[0]
    print("Bucket Number: ",''.join(query_dot.astype(str)))

    
    # Write data to a text file
    file_name = ''.join(query_dot.astype(str)) + ".txt"
    file_path = os.path.join(folder_name, file_name)    
    try:
        read_data = np.loadtxt(file_path,dtype=int)
    except: 
        print("Query doesn't match any existing buckets.")
    print(read_data)
    
    
    
    
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
    return query_dot

# # Write data to a text file
# file_path = "../random_data.txt"
# read_data = np.loadtxt(file_path)
# plane_norms = LSH(read_data, 8)
# # query = np.random.random((1, 70))
# # print(query)
# query=[read_data[0]]
# print(query)
# folder_name = "bucket_files"
# result = semantic_query_lsh(query, plane_norms,folder_name)
# print(result)
