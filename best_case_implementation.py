from typing import Dict, List, Annotated
import numpy as np
from utils import empty_folder
from Modules.LSH import *
import struct

class VecDBBest:
    def __init__(self,file_path="./DataBase/data.bin", database_path = "./DataBase", new_db = True) -> None:
        '''
        Constructor
        '''
        self.file_path =file_path # Data File Path
        self.database_path= database_path  # Path of the Folder to Create Indexes

        if new_db:
            # If New Data Base
            # Empty DataBase Folder
            empty_folder(self.database_path)
            
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
        
    def calculate_offset(self, record_id: int) -> int:
        # Calculate the offset for a given record ID
        record_size = struct.calcsize("I70f")
        return (record_id) * record_size

    def insert_records_binary(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "ab") as fout:  # Open the file in binary mode for appending
            for row in rows:
                id, embed = row["id"], row["embed"]
                # Pack the data into a binary format
                data = struct.pack(f"I{70}f", id, *embed)
                fout.write(data)
        self._build_index()

    def read_multiple_records_by_id(self, records_id: List[int]):
        record_size = struct.calcsize("I70f")
        records = {}

        with open(self.file_path, "rb") as fin:
            for i in range(len(records_id)):
                offset = self.calculate_offset(records_id[i])
                fin.seek(offset)  # Move the file pointer to the calculated offset
                data = fin.read(record_size)
                if not data:
                    records[records_id[i]] = None
                    continue

                # Unpack the binary data into a dictionary
                unpacked_data = struct.unpack("I70f", data)
                id_value, floats = unpacked_data[0], unpacked_data[1:]

                # Create and return the record dictionary
                record = {"id": id_value, "embed": list(floats)}
                records[records_id[i]] = record
        return records

    def get_top_k_records(self,k):
        records = []
        record_size = struct.calcsize("I70f")
        with open(self.file_path,'rb') as fin:
            fin.seek(0)
            for i in range(k):
                data = fin.read(record_size)
                unpacked_data = struct.unpack("I70f", data)
                id_value, floats = unpacked_data[0], unpacked_data[1:]

                record = {"id": id_value, "embed": list(floats)} 
                records.append(record)
            return records

    def _build_index(self, Level_1_nbits=8, Level_2_nbits=3, Level_3_nbits=3)-> None:
    
        '''
        Build the Index
        '''
        
        # Layer 1 Indexing
        # TODO: Here we are reading the whole file: Change later
        level_1_in = self.get_top_k_records(10000)
        level_1_planes = LSH_index(data=level_1_in, nbits=Level_1_nbits, index_path=self.database_path + "/Level1")
        np.save(self.database_path + "/Level1/"+'metadata.npy',level_1_planes)



        # Layer 2 Indexing
        for file_name in os.listdir(self.database_path + "/Level1"):
            file_path = os.path.join(self.database_path + "/Level1", file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
                read_data_2 = np.loadtxt(file_path, dtype=int, ndmin=1)
                level_2_in = self.read_multiple_records_by_id(read_data_2)
                level_2_planes = LSH_index(data=level_2_in.values(), nbits=Level_2_nbits, index_path=self.database_path + "/Level2/" + file_name[:-4])
                np.save(self.database_path + "/Level2/" + file_name[:-4]+'/metadata.npy',level_2_planes)

        # Layer 3 Indexing
        for folder_name in os.listdir(self.database_path + "/Level2"):
            folder_path = os.path.join(self.database_path + "/Level2", folder_name)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
                    read_data_3 = np.loadtxt(file_path, dtype=int, ndmin=1)
                    level_3_in = self.read_multiple_records_by_id(read_data_3)
                    level_3_planes = LSH_index(data=level_3_in.values(), nbits=Level_3_nbits, index_path=self.database_path + "/Level3/" + folder_name + '/' + file_name[:-4])
                    np.save(self.database_path + "/Level3/" + folder_name + '/' + file_name[:-4]+'/metadata.npy',level_3_planes)


    def retrive(self, query:Annotated[List[float], 70], top_k = 5)-> [int]:
        '''
        Get the top_k vectors similar to the Query

        return:  list of the top_k similar vectors Ordered by Cosine Similarity
        '''

        level_1_planes = np.load(self.database_path + "/Level1"+'/metadata.npy')
        bucket_1,result_1 = semantic_query_lsh(query, level_1_planes, self.database_path + "/Level1")
        
        if len(result_1) < top_k:
            print('level 1 smaller than top_k')

        # Retrieve from Level 2
        level_2_planes = np.load(self.database_path + "/Level2/"+bucket_1+'/metadata.npy')
        bucket_2,result_2 = semantic_query_lsh(query, level_2_planes, self.database_path + "/Level2/"+bucket_1)

        if len(result_2) < top_k:
            print('level 2 smaller than top_k')

        # Retrieve from Level 3
        level_3_planes = np.load(self.database_path + "/Level3/"+bucket_1+'/'+bucket_2+'/metadata.npy')
        bucket_3,result_3 = semantic_query_lsh(query, level_3_planes, self.database_path + "/Level3/"+bucket_1+'/'+bucket_2)

        if len(result_3) < top_k:
            print('level 3 smaller than top_k')


        index_result_3= self.read_multiple_records_by_id(result_3)

        level3_res_vectors=np.array([entry['embed'] for entry in index_result_3.values()])
        
        top_result,_=get_top_k_similar(query,level3_res_vectors,top_k)
        return top_result

