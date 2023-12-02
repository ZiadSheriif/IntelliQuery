from typing import Dict, List, Annotated
import numpy as np
from utils import empty_folder
from api import DataApi
import struct

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


    def _build_index(self) -> None:
        '''
        Build the Index
        '''

        # Level(1)
        # VIP: Don't Delete Level(1) Directory

        # Level(2)
        # VIP: Don't Delete Level(2) Directory
        
        # Level(3)
        # VIP: Don't Delete Level(3) Directory
        
        pass


    
    def retrive(self, query: Annotated[List[float], 70], top_k = 5) -> [int]:
        '''
        Get the top_k vectors similar to the Query

        return:  list of the top_k similar vectors Ordered by Cosine Similarity
        '''

        # Last Level Get the most top_k similar by cosine similarity
        pass

        return []