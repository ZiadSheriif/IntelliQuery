import numpy as np
from sklearn.cluster import KMeans
import os

from typing import Dict, List, Annotated
import struct
from utils import *

def append_to_pq_data_file(file_path, rows: List[Dict[int, Annotated[List[float], [int]]]],D_=10):
    '''
    file_path: path to binary file which will contain {id:'','PQ':}
    D_: PQ Dimension
    '''
    with open(file_path, "ab") as fout:  # Open the file in binary mode for appending
        for i,row in enumerate(rows):
            # id, pq = row["id"], row["pq"]
            id, pq = i+1, row
            # Pack the data into a binary format
            data = struct.pack(f"I{D_}I", id, *pq)
            fout.write(data)

def calculate_offset( record_id: int,D_=10) -> int:
        # Calculate the offset for a given record ID
        record_size = struct.calcsize(f"I{D_}I")
        return (record_id) * record_size

def read_multiple_pqs_by_id(file_path, records_id: List[int],D_=10):
        record_size = struct.calcsize(f"I{D_}I")
        records = {}

        with open(file_path, "rb") as fin:
            for i in range(len(records_id)):
                offset = calculate_offset(records_id[i],D_=D_)
                fin.seek(offset)  # Move the file pointer to the calculated offset
                data = fin.read(record_size)
                if not data:
                    records[records_id[i]] = None
                    continue

                # Unpack the binary data into a dictionary
                unpacked_data = struct.unpack(f"I{D_}I", data)
                id_value, floats = unpacked_data[0], unpacked_data[1:]
                # print(id_value,list(floats))

                # Create and return the record dictionary
                # record = {"id": id_value, "embed": list(floats)}
                # records[records_id[i]] = record
        return None

def read_range_binary_file_chunk(file_path,chunk_num,chunk_size=10,D_=10):
    '''
    NB:InComplete Chucks at the end are handled :D
    chunk_num: no of chunk to be read 0_indexed
    chunk_size: no of records to be retrieved
    '''
      
    record_size = struct.calcsize(f"I{D_}f")  # Assuming "I70f" format for your records

    # Open the binary file for reading
    with open(file_path, "rb") as fin:
        offset = calculate_offset(chunk_num*chunk_size,D_=D_)
        fin.seek(offset)  # Move the file pointer to the calculated offset

        # Read a chunk of records
        chunk_data = fin.read(record_size * chunk_size)

        # Break the loop if no more data is available
        if not chunk_data:
            return None #Out of index file

        # Determine the number of complete records in the chunk
        complete_records = len(chunk_data) // record_size
        remaining_bytes = len(chunk_data) % record_size

        # Process the complete records
        # Unpack the binary data into a dictionary
        records={}
        for i in range(0, complete_records * record_size, record_size):
            unpacked_data=struct.unpack(f"I{D_}I", chunk_data[i:i + record_size])
            id_value, floats = unpacked_data[0], unpacked_data[1:]
            # Create and return the record dictionary
            records[id_value]= list(floats)
            
    return records
         
def generate_pq_centroids(index_path,data,D,D_,centroids_count,n_iterations):
        # Step(1): Split Vector into sub_vectors
        sub_vectors_size=D//D_ # the Size of the sub_vector

        # If index Folder Doesn't Exist just Create it :D
        if not os.path.exists(index_path):
            os.makedirs(index_path)

        all_vectors=np.vstack([data_i['embed'] for data_i in data ])

        pq=np.zeros((len(data),D_),dtype=int)

        for  i,sub_vector_i in enumerate(range(0,D,sub_vectors_size)):
            # For each sub_vector
            # # Create Folder
            # if not os.path.exists(index_path+f"/sub_vec_{i}"):
            #     os.makedirs(index_path+f"/sub_vec_{i}")

            # Initialize KMeans
            kmeans = KMeans(n_clusters=centroids_count, n_init=n_iterations,random_state=42)  # You can set different parameters based on your requirements
            
            # Fit the KMeans model to your data
            # print(all_vectors[:,sub_vector_i:sub_vector_i+sub_vectors_size].shape)
            kmeans.fit(all_vectors[:,sub_vector_i:sub_vector_i+sub_vectors_size])

            # Get the cluster assignments and centroids
            # cluster_assignments = kmeans.labels_
            pq[:,i]=kmeans.labels_
            centroids = kmeans.cluster_centers_

            # Save Centroids to Index
            np.save(index_path+f'/sub_vec_{i}_kmeans.npy',centroids)
            # print(cluster_assignments)

        return pq


def PQ_index(data, D_=10,centroids_n_bits=3,n_iterations=5,index_path=None, D=70,generate_centroids=False):
    '''
    D_:int  the new Dimension
    centroids_n_bits: # of centroids for each sub vector is 2**centroids_n_bits
    n_iterations: # of iterations run by kmeans
    '''

    assert D%D_==0, "D_ new dimension doesn't divide the D equally"

    # Step(1): Split Vector into sub_vectors
    sub_vectors_size=D//D_ # the Size of the sub_vector

   
    # Step(2) Create Centroids
    # NB:The number of created centroids k is usually chosen as a power of 2 for more efficient memory usage. 
    centroids_count=2**centroids_n_bits

    if(generate_centroids):
        pq=generate_pq_centroids(index_path,data,D,D_,centroids_count,n_iterations)
       
    else:
        pq=np.zeros((len(data),D_),dtype=int)
        data=extract_embeds_array(data)
        for i,sub_vector_i in enumerate(range(0,D,sub_vectors_size)):

            # Read Previously created Centroids
            centroids = np.load(index_path+f'/sub_vec_{i}_kmeans.npy')

            # Create a KMeans model with the loaded centroids
            kmeans = KMeans(n_clusters=len(centroids), init=centroids, n_init=1, max_iter=1)

            # Predict cluster labels for the new data
            labels = kmeans.fit_predict(data[:,sub_vector_i:sub_vector_i+sub_vectors_size])

            pq[:,i]=labels
        print(",,,,,,,,,,,,,,,,",pq)

       

    
    # Step(3) Encode Vector
    # print(">>>>>>>",pq)
    append_to_pq_data_file(index_path+'/pq.bin',pq)

  
    return None

def distance_q_centroids(q_sub_vec,centroids):
    '''
    Computes distance between q and all centroids
    '''
    # Compute the Euclidean distances between 'vector' and each centroid
    distances = np.linalg.norm(centroids - q_sub_vec, axis=1)
    return distances

def semantic_query_pq(query,top_k, D_=10,centroids_n_bits=3,index_path=None, D=70):
    # print("semantic_query_pq")
    query=query.squeeze()
    
    sub_vectors_size=D//D_ # the Size of the sub_vector

    # Get partial distances Matrix distance between Q sub_vectors and all centroids D_x2**centroids_n_bits  [8x10]
    partial_distances=np.zeros((2**centroids_n_bits,D_))

    for  i,sub_vector_i in enumerate(range(0,D,sub_vectors_size)):
        # print(sub_vector_i)
        # continue
        # Read the Centroids
        centroids = np.load(index_path+f'/sub_vec_{i}_kmeans.npy')

        # Partial_distance for sub_vector_i
        distances=distance_q_centroids(sub_vector_i,centroids)
        partial_distances[:,i]=distances

    # print("__________________________________________")
    # print(partial_distances)
    # print(partial_distances.shape)
    # print("__________________________________________")
    
    # Loop over Data but in PQ Dimension and get the most similar [Of Course We Wan't loop over the 20M Just part]
    chunk_size=10
    scores=[]

    for i in range(0,10000000): #TODO Handle this is wrong
        records_pq=read_range_binary_file_chunk(index_path+'/pq.bin',chunk_num=i,chunk_size=chunk_size,D_=D_)
        if(records_pq is None):
            # End of File no need to continue
            break
        # For every PQ compute distance between it and the query approximately by centroids
        # records_pq {id:[pq]}
        for id,pq in records_pq.items():
            # print("..,",id)
            # print("..,",pq)
            # Get Partial Distance from the pq and the centroid
            partial_distance=partial_distances[pq,range(len(pq))]

            # Computing distance from a query to database vector by using PQ code and distance table
            query_db_vector_distance= np.sqrt(np.sum((partial_distance)**2))

            # Add this to the scores one
            scores.append((query_db_vector_distance, id))

   
    # here we assume that if two rows have the same score, return the lowest ID
    # TODO Handle if less than top_k 
    scores = sorted(scores, reverse=True)[:top_k]
    # print("scores",scores)
    return [s[1] for s in scores] 
