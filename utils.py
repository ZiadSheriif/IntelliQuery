import numpy as np
import shutil
import os
import math

import struct

def read_binary_file_chunk(file_path,record_format,start_index,chunk_size=10):
    '''
    This Function Reads Chunk from a binary File 
    If remaining from file are < chunk size they are returned normally 

    file_path:Path of the file to be read from
    record_format: format of the record ex:f"4I" 4 integers 
    start_index: index of the record from which we start reading [0_indexed]
    chunk_size: no of records to be retrieved

    @return : None in case out of index of file
              the records
    '''

    # Calculate record size 
    record_size = struct.calcsize(record_format)


    # Open the binary file for reading
    with open(file_path, "rb") as fin:
        fin.seek(start_index*record_size)  # Move the file pointer to the calculated offset

        # Read a chunk of records
        # .read() moves the file pointer (cursor) forward by the number of bytes read.
        chunk_data = fin.read(record_size * (chunk_size))
        if(len(chunk_data)==0):
            print("Out Of File Index 🔥🔥")
            return None

        # file_size = os.path.getsize(file_path)
        # print("Current file position:", fin.tell())
        # print("File size:", file_size,"record_format",record_format,"record_size",record_size,"chunk_data len",len(chunk_data))

        # Unpack Data
        records = []
        for i in range(0, len(chunk_data), record_size):
            unpacked_record =struct.unpack(record_format, chunk_data[i:i + record_size])
            id,vector=unpacked_record[0],unpacked_record[1:]
            record={"id":id,"embed":list(vector)}
            records.append(record)
        return records


def empty_folder(folder_path):
    '''
    Function to Empty a folder given its path
    @param folder_path : path of the folder to be deleted
    '''
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error while deleting {file_path}: {e}")
    print("Deleted",folder_path,"successfully")

def extract_embeds(dict):
    # {505: {'id': 505, 'embed': [0.8,....]}} --> [[0.8,....],[.......]]
    return [entry['embed'] for entry in dict.values()]

def extract_embeds_array(arr):
    return np.array([entry['embed'] for entry in arr])



def _cal_score(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity










# def generate_random(k=100):
#     # Sample data: k vectors with 70 features each
#     data = np.random.uniform(-1, 1, size=(k, 70))

#     # Write data to a text file
#     file_path = "../DataBase/random_data_"+str(k)+".txt"
#     np.savetxt(file_path, data)

#     # Read Data from File
#     # read_data = np.loadtxt(file_path)



# def array_to_dictionary(values,keys=None):
#     '''
#     values: [array of values]
#     Keys: [array of Keys] optional if not passed the keys are indexed 0-N
#     '''
#     if(keys is None):
#         keys=range(0,len(values))

#     if(len(values)!=len(keys)):
#         print ("array_to_dictionary(): InCorrect Size of keys and values")
#         return  None
    
#     dictionary_data = dict(zip(keys, values))
#     return dictionary_data


# def get_vector_from_id(data_path,id):
#     '''
#     function to get the vector by its id [BADDDDDDDD Use Seek]

#     '''
#     read_data = np.loadtxt(data_path)
#     return read_data[id]




# def check_dir(path):
#    if os.path.exists(path):
#     shutil.rmtree(path, ignore_errors=True, onerror=lambda func, path, exc: None)
#     os.makedirs(path)


# Test generate_random()
# generate_random(10000)