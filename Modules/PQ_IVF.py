from utils import *
from sklearn.cluster import MiniBatchKMeans


class PQ_IVF:
    def __init__(self,file_path,chunk_size,K_means_n_clusters,K_means_max_iter,ivf_folder_path,pq_D_,pq_K_means_n_clusters):
        '''
        file_path:Path to .bin file to of raw data to apply PQ as generated by insert binary function in API class
        chunk_size: no of records to be processing together in while performing kmeans
        K_means_n_clusters: No of Kmeans Clusters
        K_means_max_iter: max iteration by kmeans default [100] in sklearn
        ivf_folder_path: path of folder to store ivf_s

        pq_D_: Dimensionality of vector to be reduced to 
        pq_K_means_n_clusters: no of centroids for the sub_vector to be used in PQ [Power of 2] 

        '''
        self.file_path=file_path  # Original Data
        self.chunk_size=chunk_size
        
        self.K_means_n_clusters=K_means_n_clusters
        self.K_means_max_iter=K_means_max_iter
        
        self.ivf_folder_path=ivf_folder_path
        assert 70%pq_D_==0, "D_ dimension doesn't divide the D equally"
        self.pq_D_=pq_D_
        assert pq_K_means_n_clusters%2==0, "# centroids k is usually chosen as a power of 2 for more efficient memory usage"
        self.pq_K_means_n_clusters=pq_K_means_n_clusters

        self.K_means_centroids=None
        # self.pq_sub_vectors_K_means_centroids=None
        

    def PQ_IVF_index(self):
        print("---PQ_IVF_index()----")
        # ############################################################### ################################# ###############################################################
        # ############################################################### Step(1):Clustering Data from file ###############################################################
        # ############################################################### ################################# ###############################################################

        # We need to Read Data from File chunk by chunk
        file_size = os.path.getsize(self.file_path)
        record_size=struct.calcsize(f"I{70}f")
        n_records=file_size/record_size
        no_chunks=math.ceil(n_records/self.chunk_size)

        # Step(1) Getting Centroids
        # Initialize MiniBatchKMeans to clustering
        kmeans = MiniBatchKMeans(n_clusters=self.K_means_n_clusters, batch_size=self.chunk_size, max_iter=self.K_means_max_iter,n_init=4,random_state=42)
        
        print("***********Reading File in build index()*****************")
        print("data.bin file size (R=284 Byte): ",file_size)
        print("n_records",n_records)
        print("chuck size",self.chunk_size)
        print("No of Chunks",no_chunks)

        # Loop to get the Kmeans Centroids
        for i in range(no_chunks):
            print("Reading Chunk",i,"To compute centroids....")
            data_chunk=read_binary_file_chunk(file_path=self.file_path,record_format=f"I{70}f",start_index=i*self.chunk_size,chunk_size=self.chunk_size) #[{"id":,"embed":[]}]
            # Extract Data
            chunk_vectors=np.array([entry['embed'] for entry in data_chunk])
            kmeans.partial_fit(chunk_vectors)
            self.K_means_centroids = kmeans.cluster_centers_
            # if(data_chunk is None):
            #     # If out of index but not needed here
            #     break



        # ############################################################### ######################################## ###############################################################
        # ############################################################### Step(2):Getting Residual for each vector ###############################################################
        # ############################################################### ################################# ###############################################################
        for i in range(no_chunks):
            print("Reading Chunk",i,"To predict ....")
            data_chunk=read_binary_file_chunk(file_path=self.file_path,record_format=f"I{70}f",start_index=i*self.chunk_size,chunk_size=self.chunk_size) #[{"id":,"embed":[]}]
            # Extract Data
            ids=np.empty(len(data_chunk),dtype=int)
            vectors=np.empty((len(data_chunk),70))
            for i,entry in enumerate(data_chunk):
                id,vector=entry['id'],entry['embed']
                ids[i]=id
                vectors[i]=vector
            
            # Get Cluster for each 1
            labels=kmeans.predict(vectors) #Each vector corresponding centroid

            # print("K_means_centroids",self.K_means_centroids)
            # print("data_chunk[0]",data_chunk[0])
            # print("ids[0]",ids[0])
            # print("labels[0]",labels[0])



            for label in set(labels):
                region_ids=ids[labels==label]  # get ids belonging to such region
                region_vectors=vectors[labels==label]  # get vectors belonging to such region
                region_vectors-=self.K_means_centroids [label]

                # if(label==labels[0]):
                #     print(region_vectors)
                #     return None
    
                # Open file of this Region(cluster)
                with open(self.ivf_folder_path+f'/residuals_{label}.bin', "ab") as fout:
                    for i in range(len(region_ids)):
                        data = struct.pack(f"I{70}f", region_ids[i], *region_vectors[i])
                        fout.write(data)


            # Test Read residuals of region 0
            # data_chunk=read_binary_file_chunk(file_path=self.ivf_folder_path+f'/residuals_{0}.bin',record_format=f"I{70}f",start_index=0,chunk_size=500) #[{"id":,"embed":[]}]



            # ############################################################### ########################################## ###############################################################
            # ############################################################### Step(3):Getting Centroids for PQ Residuals ###############################################################
            # ############################################################### ########################################## ###############################################################
            sub_vectors_size=70//self.pq_D_ # the Size of the sub_vector


            # Initialize MiniBatchKMeans to clustering
            sub_vectors_kmeans = [MiniBatchKMeans(n_clusters=self.pq_K_means_n_clusters, batch_size=self.chunk_size, max_iter=self.K_means_max_iter,n_init=4,random_state=42) for _ in  range(self.pq_D_) ]
            # print(len(sub_vectors_kmeans))
        
            # (1) Generating  centroids of each sub_vector
            for label in range(self.K_means_n_clusters):
                # Loop over each cluster
                residual_file_path=self.ivf_folder_path+f'/residuals_{label}.bin'
                file_size = os.path.getsize(residual_file_path)
                record_size=struct.calcsize(f"I{70}f")
                n_records=file_size/record_size
                no_chunks=math.ceil(n_records/self.chunk_size)

                for i in range(no_chunks):
                    # We need to Read Residuals from File chunk by chunk
                    print("Reading Region",label ,"Chunk",i,"to compute centroids ....")
                    data_chunk=read_binary_file_chunk(file_path=residual_file_path,record_format=f"I{70}f",start_index=i*self.chunk_size,chunk_size=self.chunk_size) #[{"id":,"embed":[]}]
                    # Extract Data
                    data_chunk_dict={} #{id:[residual]}
                    for entry in data_chunk:
                        id,vector=entry['id'],entry['embed']
                        data_chunk_dict[id]=vector

                    # print( np.array(list(data_chunk_dict.keys())))
                    # print( np.array(list(data_chunk_dict.values())))
                    vectors_res=np.array(list(data_chunk_dict.values()))   # dim: n*70

                    # Split into sub_vectors
                    for j,sub_vector_start_index in enumerate(range(0,70,sub_vectors_size)):
                        # print(sub_vector_i)
                        # print(chunk_vectors.shape)
                        # print(np.array(data_chunk_dict.values()).shape)
                        sub_vectors_kmeans[j].partial_fit(vectors_res[:,sub_vector_start_index:sub_vector_start_index+sub_vectors_size])


            # Save Centroids for each sub_vector
            ###########################################
            # We have sub_vectors_kmeans   kmeans for each sub_vector for all clusters together
            ############################################
            for i,kmeans_obj in enumerate(sub_vectors_kmeans):
                # print(kmeans_obj.cluster_centers_.shape) # dim self.pq_K_means_n_clusters * sub_vectors_size #[TODO Check if need to save ]
                np.save(self.ivf_folder_path+f'/sub_vec_{i}_kmeans.npy',kmeans_obj.cluster_centers_)
            
            # ############################################################### ###################################### ###############################################################
            # ############################################################### Step(4):Getting PQ for every Residuals ###############################################################
            # ############################################################### ###################################### ###############################################################
            for label in range(self.K_means_n_clusters):
                # Loop over each cluster
                residual_file_path=self.ivf_folder_path+f'/residuals_{label}.bin'
                file_size = os.path.getsize(residual_file_path)
                record_size=struct.calcsize(f"I{70}f")
                n_records=file_size/record_size
                no_chunks=math.ceil(n_records/self.chunk_size)

                for i in range(no_chunks):
                    # We need to Read Residuals from File chunk by chunk
                    print("Reading Region",label ,"Chunk",i,"to Predict ....")
                    data_chunk=read_binary_file_chunk(file_path=residual_file_path,record_format=f"I{70}f",start_index=i*self.chunk_size,chunk_size=self.chunk_size) #[{"id":,"embed":[]}]
                    # Extract Data
                    data_chunk_dict={} #{id:[residual]}
                    for entry in data_chunk:
                        id,vector=entry['id'],entry['embed']
                        data_chunk_dict[id]=vector
                    
                    vectors_res=np.array(list(data_chunk_dict.values()))   # dim: n(no of vectors)*70
                    ids_res=np.array(list(data_chunk_dict.keys()))  # dim:n(no of vectors)


                    pq__of_chunk_i=np.zeros((vectors_res.shape[0],self.pq_D_),dtype=int)
                    # print(pq__of_chunk_i.shape) #dim n*10  -> sum(n)=no of records

                    # Split into sub_vectors
                    for j,sub_vector_start_index in enumerate(range(0,70,sub_vectors_size)):
                        # Loop = no of sub_vectors (D_)
                        n_sub_vector=vectors_res[:,sub_vector_start_index:sub_vector_start_index+sub_vectors_size] #dim: n(no of vectors)*sub_vectors_size
                        labels=sub_vectors_kmeans[j].predict(n_sub_vector) # [1 0 4 9 ] #the centroid of each vector NB they are 1 sub_vector not all sub_vectors 
                        # print(labels)
                        pq__of_chunk_i[:,j]=labels

                    # print("pq__of_chunk_i",pq__of_chunk_i) # dim: n(no of sub_vectors)*10
                    # print("ids_res",ids_res)

                    # Save PQ
                    with open(self.ivf_folder_path+f'/residuals_{label}_pq.bin', "ab") as fout:  # Open the file in binary mode for appending
                        for i,row in enumerate(pq__of_chunk_i):
                            # print(ids_res[i], *pq__of_chunk_i[i])   id & pq
                            data = struct.pack(f"I{self.pq_D_}I", ids_res[i], *pq__of_chunk_i[i])
                            fout.write(data)

            
            return
  

    def semantic_query_pq_ivf(self,query,top_k,n_regions):
        '''
        top_k: nearest 
        n_regions:no of regions to be candidates of the clusters
        '''
        # Use Centroid TODO Check if we need to not save just read
        print("semantic_query_pq_ivf ()")

            
        # ################################################### ########################################################################## ###############################################################
        # ################################################### Step(1) for query ,the k nearest centroids of Voronoi partitions are found ###############################################################
        # ################################################### ########################################################################## ###############################################################
        
        assert self.K_means_centroids.shape[0]>n_regions,"Top K must be less than the no of regions [Check Medium]"
        # Calculate distances using Euclidean distance (you can also use cosine similarity) [TODO check]
        distances = np.linalg.norm(self.K_means_centroids - query, axis=1)
        # self.K_means_centroids.predict(data)

        # Get indices of the nearest vectors
        nearest_regions= np.argsort(distances)[:n_regions]

        # Get the nearest centroid
        nearest_centroids = self.K_means_centroids[nearest_regions]
        # print("nearest_centroid",nearest_centroids.shape,nearest_centroids)


             
        # ################################################### ####################################################################### ###############################################################
        # ################################################### Step(2) calculate query residual separately for every Voronoi partition ###############################################################
        # ################################################### ####################################################################### ###############################################################
        query_residual=np.empty((nearest_centroids.shape[0],70))
        for i in range(nearest_centroids.shape[0]):
            # For Every Separate Region
            # Get Residual of query to region's centroid
            query_residual[i]=nearest_centroids[i]-query
            # print("query_residual",query_residual.shape,query_residual)


        # ################################################### ##################################################################################### ###############################################################
        # ################################################### Step(3) split the query residual is then split into sub_vectors& fet partial Distance ###############################################################
        # ################################################### ##################################################################################### ###############################################################
        # Get partial distances Matrix distance between Q sub_vectors and all centroids
        partial_distances_region_i=np.empty((self.pq_K_means_n_clusters,self.pq_D_,nearest_centroids.shape[0])) # #clusters for the sub_vector*no of sub_vectors*no of regions
        # partial_distances_region_i=np.zeros((self.pq_K_means_n_clusters,self.pq_D_,2)) # #clusters for the sub_vector*no of sub_vectors*no of regions

        sub_vectors_size=70//self.pq_D_ # the Size of the sub_vector
        for j,sub_vector_start_index in enumerate(range(0,70,sub_vectors_size)):
            # Loop over each sub vector
            # Read the Centroids for this sub_vector
            sub_vector_centroids= np.load(self.ivf_folder_path+f'/sub_vec_{j}_kmeans.npy')
            # print(sub_vector_centroids.shape) #self.pq_K_means_n_clusters * pq_D_
            for region_i in range(nearest_centroids.shape[0]):
                # For Each Region
                # Compute distance from the query to all the centroids
                distances = np.linalg.norm(sub_vector_centroids - query_residual[region_i,sub_vector_start_index:sub_vector_start_index+sub_vectors_size] , axis=1)
                # print(distances)
                # print(sub_vector_centroids.shape)
                # print(query_residual.shape)
                # print(distances.shape)
                partial_distances_region_i[:,j,region_i]=distances
        # print(partial_distances_region_i[:,:,0])
        # print(partial_distances_region_i[:,:,1])


        # ########################################### ####################################################################################################### ###############################################################
        # ########################################### Step(4) Get approximate Distance between each candidate and the query using the PQ and partial distance ###############################################################
        # ########################################### ####################################################################################################### ###############################################################
        # top_k=np.ones() # TODO instead of sorting all of them just pick the largest k
        scores=[]
        for region_i,region in enumerate(nearest_regions):
            # Every Region
            file_size = os.path.getsize(self.ivf_folder_path+f'/residuals_{region}_pq.bin')
            record_size=struct.calcsize(f"I{self.pq_D_}I")
            n_records=file_size/record_size
            no_chunks=math.ceil(n_records/self.chunk_size)
            # print(region,n_records)

            for i in range(no_chunks):
                # Read the residual chunk by chunk
                data_chunk=read_binary_file_chunk(file_path=self.ivf_folder_path+f'/residuals_{region}_pq.bin',record_format=f"I{self.pq_D_}I",start_index=i*self.chunk_size,chunk_size=self.chunk_size) #[{"id":,"embed":[PQ]}]
                # print(data_chunk) #PQ of the elements in region
                for entry in data_chunk:
                        id,pq=entry['id'],entry['embed']
                        # print(id,pq)
                        # Slice elements based on the indices [Use arange ]
                        partial_distance = partial_distances_region_i[pq, np.arange(partial_distances_region_i.shape[1]),region_i]
                        
                        # print(partial_distance)
                        # partial_distance=partial_distances_region_i[pq,:,region_i]
                        # print(partial_distance.shape)
                        # print(partial_distance)

                        
                        # Computing distance from a query to database vector by using PQ code and distance table
                        query_db_vector_distance= np.sqrt(np.sum((partial_distance)**2))
                        # print("query_db_vector_distance",query_db_vector_distance)

                        # Add this to the scores one
                        scores.append((query_db_vector_distance, id))
        
        # ########################################### ##################### ###################################################
        # ########################################### Step(5) Get nearest k ###################################################
        # ########################################### ##################### ###################################################
        # TODO Handle if less than top_k 
        # here we assume that if two rows have the same score, return the lowest ID
        scores = sorted(scores, reverse=True)[:top_k]

        return [s[1] for s in scores] 