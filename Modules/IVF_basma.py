from utils import *
from sklearn.cluster import MiniBatchKMeans
# from utils import _cal_score



def IVF_index(file_path,K_means_n_clusters,k_means_batch_size,k_means_max_iter,k_means_n_init,chunk_size,index_folder_path):
    '''
    file_path: path to the data .bin file

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

    kmeans = MiniBatchKMeans(n_clusters=K_means_n_clusters, batch_size=k_means_batch_size, max_iter=k_means_max_iter,n_init=k_means_n_init,random_state=42)


    # We need to Read Data from File chunk by chunk
    file_size = os.path.getsize(file_path)
    record_size=struct.calcsize(f"I{70}f")
    n_records=file_size/record_size
    no_chunks=math.ceil(n_records/chunk_size)

    # Step(1) Getting centroids:
    # Loop to get the Kmeans Centroids
    for i in range(no_chunks):
        data_chunk=read_binary_file_chunk(file_path=file_path,record_format=f"I{70}f",start_index=i*chunk_size,chunk_size=chunk_size) #[{"id":,"embed":[]}]
        # TODO Remove this loop
        chunk_vectors=np.array([entry['embed'] for entry in data_chunk])
        kmeans.partial_fit(chunk_vectors)

    # Centroids
    K_means_centroids=kmeans.cluster_centers_
    # Saving Centroids #TODO Check precision of centroids after read and write in the file @Basma Elhoseny 
    write_binary_file(file_path=index_folder_path+'/centroids.bin',data_to_write=K_means_centroids,format=f"{70}f")

    # ##################################################################
    # #TEST# Centroids are Written Correct #############################
    # ##################################################################
    # K_means_centroids_check=read_binary_file(index_folder_path+'/centroids.bin',f"{70}f")
    # print("Note: Some Approximations happen due to float ")
    # print(K_means_centroids_check[2])
    # print(K_means_centroids[2])




    # Step(2) Getting vectors of each regions
    for i in range(no_chunks):
        print("IVF_indexing Chunk",i,".......")
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
                    data = struct.pack(f"I{70}f", region_ids[i], *region_vectors[i])
                    fout.write(data)

    
            
                
    ###################################################################
    ##TEST# no. of elements in each cluster ###########################
    ###################################################################
    # count_region=[]
    # for region in range(K_means_n_clusters):
    #     with open(index_folder_path+f'/cluster{region}.bin', "ab") as fout:
    #             file_size = os.path.getsize(index_folder_path+f'/cluster{region}.bin')
    #             record_size=struct.calcsize(f"I{70}f")
    #             n_records=file_size/record_size

    #             count_region.append(n_records)

    # print(count_region) 
    # print(np.sum(np.array(count_region)))
    return


def semantic_query_pq_ivf(query,top_k):
    '''
    top_k: nearest 
    n_regions:no of regions to be candidates of the clusters
    '''
    # Use Centroid TODO Check if we need to not save just read
    print("semantic_query_ivf ()")

    return 