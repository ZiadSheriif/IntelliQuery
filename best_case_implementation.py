import numpy as np
import os
from utils import empty_folder
from api import DataApi
from Modules.LSH import  *
from typing import Dict, List, Annotated
import numpy as np
from utils import empty_folder
from api import DataApi

class VecDBBest:
    def __init__(self,file_path="./DataBase/data.bin", database_path = "./DataBase", new_db = True) -> None:
        '''
        Constructor
        '''
        self.file_path =file_path # Data File Path
        self.database_path= database_path  # Path of the Folder to Create Indexes
        self.data_api= DataApi(file_path)

        if new_db:
            # If New Data Base
            # Empty DataBase Folder
            empty_folder(self.database_path)
            
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
        
    def insert_records_binary(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
            self.data_api.insert_records_binary(rows)
            # with open(self.file_path, "ab") as fout:  # Open the file in binary mode for appending
            #     for row in rows:
            #         id, embed = row["id"], row["embed"]
            #         # Pack the data into a binary format
            #         data = struct.pack(f"I{70}f", id, *embed)
            #         fout.write(data)
            self._build_index()
            
    def _build_index(self, Level_1_nbits=8, Level_2_nbits=3, Level_3_nbits=3)-> None:
    
        '''
        Build the Index
        '''
        
        # Layer 1 Indexing
        level_1_in = self.data_api.get_top_k_records(10000)
        self.level_1_planes = LSH_index(data=level_1_in, nbits=Level_1_nbits, index_path=self.database_path + "/Level1")

        # Layer 2 Indexing
        self.level_2_planes = {}
        for file_name in os.listdir(self.database_path + "/Level1"):
            file_path = os.path.join(self.database_path + "/Level1", file_name)
            if os.path.isfile(file_path):
                read_data_2 = np.loadtxt(file_path, dtype=int, ndmin=1)
                level_2_in = self.data_api.get_multiple_records_by_ids(read_data_2 - 1)
                self.level_2_planes[file_name[:-4]] = LSH_index(data=level_2_in.values(), nbits=Level_2_nbits, index_path=self.database_path + "/Level2/" + file_name[:-4])

        # Layer 3 Indexing
        self.level_3_planes = {}
        for folder_name in os.listdir(self.database_path + "/Level2"):
            self.level_3_planes[folder_name] = {}
            folder_path = os.path.join(self.database_path + "/Level2", folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    read_data_3 = np.loadtxt(file_path, dtype=int, ndmin=1)
                    level_3_in = self.data_api.get_multiple_records_by_ids(read_data_3)
                    self.level_3_planes[folder_name][file_name[:-4]] = LSH_index(data=level_3_in.values(), nbits=Level_3_nbits, index_path=self.database_path + "/Level3/" + folder_name + '/' + file_name[:-4])

    def retrieve(self, query:Annotated[List[float], 70], top_k = 5,level=1)-> [int]:
        '''
        Get the top_k vectors similar to the Query

        return:  list of the top_k similar vectors Ordered by Cosine Similarity
        '''
        if level == 1:
            # Retrieve from Level 1
            bucket_1,result_1 = semantic_query_lsh(query, self.level_1_planes, self.database_path + "/Level1")
        elif level == 2:
            # Retrieve from Level 2
            bucket_2,result_2 = semantic_query_lsh(query, self.level_2_planes[bucket_1], self.database_path + "/Level2")
        elif level == 3:
            # Retrieve from Level 3
            bucket_3,result_3 = semantic_query_lsh(query, self.level_3_planes[bucket_1][bucket_2], self.database_path + "/Level3")
        else:
            index_result_3= self.data_api .get_multiple_records_by_ids(result_3)
            level3_res_vectors=[entry['embed'] for entry in index_result_3.values()]
            top_result,_=get_top_k_similar(query,level3_res_vectors,10)

        return top_result

