# import files
from utils import *
from sklearn.cluster import KMeans

# improt libraries
import heapq
import numpy as np


def IVF_index(file_path,K_means_metric,K_means_n_clusters,k_means_batch_size,k_means_max_iter,k_means_n_init,chunk_size,index_folder_path):
    '''
    file_path: path to the data .bin file

    K_means_metric: metric to be used in clustering cosine or euclidean' or TODO use SCANN idea ERORR Think of another way this isn't supported in kmeans
    K_means_n_clusters: No of Kmeans Clusters
    k_means_batch_size: kmeans batch size to be sampled at each iteration of fitting
    k_means_max_iter: max iteration by kmeans default [100] in sklearn
    k_means_n_init:The number of times the algorithm will be run with different centroid seeds.

    chunk_size: chunk_size: no of records to be processing together in while performing kmeans

    ivf_folder_path: Folder path to store regions of kmeans
    '''
    print("---IVF_index()----")
    # ############################################################### ################################# ###############################################################
    # ############################################################### Step(1):Clustering Data from file ###############################################################
    # ############################################################### ################################# ###############################################################
    kmeans = KMeans(n_clusters=K_means_n_clusters, max_iter=k_means_max_iter,n_init=k_means_n_init,random_state=42)
    

    # Use the first Chunck to only get teh centroids
    data_chunk=read_binary_file_chunk(file_path=file_path,record_format=f"I{70}f",start_index=0,chunk_size=1000000) #[{"id":,"embed":[]}]
    # TODO Remove this loop
    chunk_vectors=np.array([entry['embed'] for entry in data_chunk])
    kmeans.fit(chunk_vectors)



    # We need to Read Data from File chunk by chunk
    file_size = os.path.getsize(file_path)
    record_size=struct.calcsize(f"I{70}f")
    n_records=file_size/record_size
    no_chunks=math.ceil(n_records/chunk_size)

    # # Step(1) Getting centroids:
    # # Loop to get the Kmeans Centroids
    # for i in range(no_chunks):
    #     data_chunk=read_binary_file_chunk(file_path=file_path,record_format=f"I{70}f",start_index=i*chunk_size,chunk_size=chunk_size) #[{"id":,"embed":[]}]
    #     # TODO Remove this loop
    #     chunk_vectors=np.array([entry['embed'] for entry in data_chunk])
    #     kmeans.partial_fit(chunk_vectors)

    # Centroids
    K_means_centroids=kmeans.cluster_centers_
    # Saving Centroids #TODO Check precision of centroids after read and write in the file @Basma Elhoseny 
    write_binary_file(file_path=index_folder_path+'/centroids.bin',data_to_write=K_means_centroids,format=f"{70}f")

    # ##################################################################
    # #TEST# Centroids are Written Correct #############################
    # ##################################################################



    # Step(2) Getting vectors of each regions
    for i in range(no_chunks):
        data_chunk=read_binary_file_chunk(file_path=file_path,record_format=f"I{70}f",start_index=i*chunk_size,chunk_size=chunk_size,dictionary_format=True) #[{109: np.array([70 dim])}]

        # Get Cluster for each one
        labels=kmeans.predict(list(data_chunk.values())) #Each vector corresponding centroid


        ids=np.array(list(data_chunk.keys()))
        vectors=np.array(list(data_chunk.values()))
        data_chunk=None  #Clear Memory

        # Add vectors to their corresponding region
        for label in set(labels):
            region_ids=ids[labels==label]  # get ids belonging to such region
            region_vectors=vectors[labels==label]  # get vectors belonging to such region
            # Open file of this Region(cluster) Just Once for every Region :D
            with open(index_folder_path+f'/cluster{label}.bin', "ab") as fout:
                for i in range(len(region_ids)):
                    #TODO Check whether store id of the vector @Basma Elhoseny
                    data = struct.pack(f"I{70}f", region_ids[i],*region_vectors[i,:])
                    fout.write(data)
    
            
            
    return



def semantic_query_ivf(data_file_path, index_folder_path, query, top_k, n_regions):
    query = np.squeeze(np.array(query))


    # Read Centroids
    K_means_centroids = read_binary_file(index_folder_path + '/centroids.bin', f"70f")     


    assert K_means_centroids.shape[0] > n_regions, "n_regions must be less than the number of regions"


    # Calculate distances to centroids
    distances = np.linalg.norm(K_means_centroids - query, axis=1)
    # Get indices of the nearest centroids
    nearest_regions = np.argsort(distances)[:n_regions]
    

    # Use a heap to keep track of the top k scores
    top_scores_heap = []
    for region in nearest_regions:
        records=read_binary_file_chunk(index_folder_path+f'/cluster{region}.bin', f'I{70}f', 0, chunk_size=100000000000,dictionary_format=True)


        # Vectorize cosine similarity calculation
        vectors = np.array([record for record in records.values()])
        dot_products = np.dot(vectors, query)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        similarities = dot_products / norms

        # Process the scores and maintain a heap
        for score, id in zip(similarities, records.keys()):
            if len(top_scores_heap) < top_k:
                heapq.heappush(top_scores_heap, (score, id))
            else:
                heapq.heappushpop(top_scores_heap, (score, id))
        
    # Sort and get the top k scores
    top_scores_heap.sort(reverse=True)
    top_k_ids = [id for _, id in top_scores_heap]
    
    return top_k_ids
