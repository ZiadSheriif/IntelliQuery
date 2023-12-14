
from typing import Dict, List, Annotated
from utils import *
import numpy as np
import time
import os
from Modules.LSH import LSH_index, semantic_query_lsh
from Modules.IVF_basma import IVF_index,semantic_query_ivf

NUMBER_OF_RECORDS_BRUTE_FORCE = 10000

class VecDB:
    def __init__(self,file_path="./DataBase", new_db = True) -> None:
            '''
            Constructor
            '''
            self.file_path =file_path+'/data.bin' # Data File Path
            self.database_path= file_path  # Path of the Folder to Create Indexes

            if new_db:
                if not os.path.exists(self.database_path):
                    os.makedirs(self.database_path)

                else: 
                  # If New DataBase Empty DataBase Folder
                  empty_folder(self.database_path)
                
                # just open new file to delete the old one
                with open(self.file_path, "w") as fout:
                    # if you need to add any head to the file
                    pass


            self.level1=None

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # Append in Binary Mode
        with open(self.file_path, "ab") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                # Pack the data into a binary format
                data = struct.pack(f"I{70}f", id, *embed)
                fout.write(data)
        self._build_index()


    def _build_index(self,Level_1_nbits=5, Level_2_nbits=3, Level_3_nbits=3,Level_4_nbits=3)-> None:
    
        '''
        Build the Index
        '''
        file_size = os.path.getsize(self.file_path)
        record_size=struct.calcsize(f"I{70}f")
        n_records=file_size/record_size
        if(n_records<100000):
            self.number_of_clusters=100
        else: 
            self.number_of_clusters=int(n_records/NUMBER_OF_RECORDS_BRUTE_FORCE)
        print("Record Size: ",record_size)
        print("File Size: ",file_size)
        print("Building Index ..........")   
        print("number_of_clusters: ",self.number_of_clusters)
        # measure the time
        start = time.time()

        # Make Level1 Folder
        Level1_folder_path = self.database_path+'/Level1'
        if not os.path.exists(Level1_folder_path):
                os.makedirs(Level1_folder_path)

        # IVF Layer 1 Indexing
        chunk_size=1000
        IVF_index(file_path=self.file_path,K_means_metric='euclidean',K_means_n_clusters=self.number_of_clusters,k_means_batch_size=chunk_size,k_means_max_iter=100,k_means_n_init='auto',chunk_size=chunk_size,index_folder_path=Level1_folder_path)
          
        
        # # Layer 1 Indexing
        # # level_1_in = self.get_top_k_records(top_k_records)
        # level_1_planes = LSH_index(file_path=self.file_path, nbits=Level_1_nbits, chunk_size=1000,index_path=self.database_path + "/Level1")
        # np.save(self.database_path + "/Level1/"+'metadata.npy',level_1_planes)
        # print("Layer 1 Finished")
        # return
        
        
        
        # # Layer 2 Indexing
        # for file_name in os.listdir(self.database_path + "/Level1"):
        #     file_path = os.path.join(self.database_path + "/Level1", file_name)
        #     if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
        #         read_data_2 = np.loadtxt(file_path, dtype=int, ndmin=1)
        #         level_2_in = self.read_multiple_records_by_id(read_data_2)
        #         level_2_planes = LSH_index(data=level_2_in.values(), nbits=Level_2_nbits, index_path=self.database_path + "/Level2/" + file_name[:-4])
        #         np.save(self.database_path + "/Level2/" + file_name[:-4]+'/metadata.npy',level_2_planes)
        # print("Layer 2 Finished")
        # return
        
        
        # # Layer 3 Indexing
        # for folder_name in os.listdir(self.database_path + "/Level2"):
        #     folder_path = os.path.join(self.database_path + "/Level2", folder_name)
        #     for file_name in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, file_name)
        #         if os.path.isfile(file_path)  and file_name.lower().endswith(".txt"):
        #             read_data_3 = np.loadtxt(file_path, dtype=int, ndmin=1)
        #             level_3_in = self.read_multiple_records_by_id(read_data_3)
        #             level_3_planes = LSH_index(data=level_3_in.values(), nbits=Level_3_nbits, index_path=self.database_path + "/Level3/" + folder_name + '/' + file_name[:-4])
        #             np.save(self.database_path + "/Level3/" + folder_name + '/' + file_name[:-4]+'/metadata.npy',level_3_planes)
        # print("Layer 3 Finished")
        # return

        # # Layer 4 Indexing
        # for folder_name in os.listdir(self.database_path + "/Level3"):
        #     folder_path = os.path.join(self.database_path + "/Level3", folder_name)
        #     for folder_name_2 in os.listdir(folder_path):
        #         folder_path_2 = os.path.join(folder_path, folder_name_2)
        #         for file_name in os.listdir(folder_path_2):
        #             file_path = os.path.join(folder_path_2, file_name)
        #             if os.path.isfile(file_path)  and file_name.lower().endswith(".txt"):
        #                 read_data_4 = np.loadtxt(file_path, dtype=int, ndmin=1)
        #                 level_4_in = self.read_multiple_records_by_id(read_data_4)
        #                 level_4_planes = LSH_index(data=level_4_in.values(), nbits=Level_4_nbits, index_path=self.database_path + "/Level4/" + folder_name + '/' + folder_name_2 + '/' + file_name[:-4])
        #                 np.save(self.database_path + "/Level4/" + folder_name + '/' + folder_name_2 + '/' + file_name[:-4]+'/metadata.npy',level_4_planes)
        # print("Layer 4 Finished")
        
        
        # measure the time
        end = time.time()
        print("Indexing Done ...... Time taken by Indexing: ",end - start)
        return 
    
    def retrive(self, query:Annotated[List[float], 70],top_k = 5)-> [int]:
        '''
        Get the top_k vectors similar to the Query

        return:  list of the top_k similar vectors Ordered by Cosine Similarity
        '''
        print(f"Retrieving top {top_k} ..........")
        Level1_folder_path = self.database_path+'/Level1'
        final_result=semantic_query_ivf(data_file_path=self.file_path,index_folder_path=Level1_folder_path,query=query,top_k=top_k,n_regions=5)

        return final_result

        
           
        # # Retrieve from Level 1
        # level_1_planes = np.load(self.database_path + "/Level1"+'/metadata.npy')
        # bucket_1,result = semantic_query_lsh(query, level_1_planes, self.database_path + "/Level1")
        # print("length of first bucket",result.shape)

        # if len(result) < top_k:
        #     print('level 1 smaller than top_k')
        
        # # Retrieve from Level 2
        # level_2_planes = np.load(self.database_path + "/Level2/"+bucket_1+'/metadata.npy')
        # bucket_2,result = semantic_query_lsh(query, level_2_planes, self.database_path + "/Level2/"+bucket_1)
        # print("length of second bucket",result.shape)

        # if len(result) < top_k:
        #     print('level 2 smaller than top_k')

        # # Retrieve from Level 3
        # level_3_planes = np.load(self.database_path + "/Level3/"+bucket_1+'/'+bucket_2+'/metadata.npy')
        # bucket_3,result = semantic_query_lsh(query, level_3_planes, self.database_path + "/Level3/"+bucket_1+'/'+bucket_2)
        # print("length of third bucket",result.shape)
        
        # if len(result) < top_k:
        #     print('level 3 smaller than top_k')
        
        # # Retrieve from Level 4
        # level_4_planes = np.load(self.database_path + "/Level4/"+bucket_1+'/'+bucket_2+'/'+bucket_3+'/metadata.npy')
        # bucket_4,result = semantic_query_lsh(query, level_4_planes, self.database_path + "/Level4/"+bucket_1+'/'+bucket_2+'/'+bucket_3)
        # print("length of fourth bucket",result.shape)

        # if len(result) < top_k:
        #     print('level 4 smaller than top_k')
        
        
        # # Retrieve from Data Base the Embeddings of the Vectors
        # final_result= read_multiple_records_by_id(self.file_path,result)
        
        # Calculate the Cosine Similarity between the Query and the Vectors
        # scores = []
        # for row in final_result.values():
        #     id_value = row['id']
        #     embed_values = row['embed']
        #     score = self._cal_score(query, embed_values)
        #     scores.append((score, id_value))
        # scores = sorted(scores, reverse=True)[:top_k]
        # return [s[1] for s in scores]
        
        
        

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity