import numpy as np
import shutil
import os
import math
from typing import Dict, List, Annotated
import struct
import sys


def save_20M_record(data):
    '''
    Given 20M record save them as required by the TA
    data: (20M,70)
    '''

    folder_name='./Data_TA'
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    empty_folder(folder_name)

    # files=['data_100K.bin',"data_1M.bin","data_5M.bin","data_10M.bin","data_15M.bin","data_20M.bin"]
    files=['data_100K.bin']
    # files=["data_10M.bin","data_15M.bin","data_20M.bin"]
    # limits=[10**5,10**6,5*10**6,10**7,15*10**6,2*10**7]
    limits=[10**5]
    # limits=[10**7,15*10**6,2*10**7]

    for i,file in enumerate(files):
        data_part=data[:limits[i]]

        # Append in Binary Mode
        with open(folder_name+'/'+file, "ab") as fout:
            for id,vector in enumerate(data_part):
                # Pack the data into a binary format
                unpacked_data = struct.pack(f"I{70}f", id, *vector)
                fout.write(unpacked_data)

    # # Test
    # print("len(data)",len(data))
    # # print(data[0])
    # print(data[-1])
    # read_data=read_binary_file_chunk('./Data_TA/data_100K.bin',f"I{70}f",start_index=0,chunk_size=1000000,dictionary_format=True)
    # print("len(read_data)",len(read_data))
    # # print(read_data[0])
    # print(read_data[10**5-1])


    # # Test
    # print("len(data)",len(data))
    # # print(data[0])
    # print(data[-1])
    # read_data=read_binary_file_chunk('./Data_TA/data_1M.bin',f"I{70}f",start_index=0,chunk_size=1000000,dictionary_format=True)
    # print("len(read_data)",len(read_data))
    # # print(read_data[0])
    # print(read_data[10**6-1])

def read_binary_file(file_path,format):
    '''
    Read binary file from its format
    '''
    try:
        with open(file_path,"rb") as fin:
            file_size = os.path.getsize(file_path)
            record_size=struct.calcsize(format)
            n_records=file_size/record_size
            # print("n_records",n_records)

            fin.seek(0) #Move pointer to the beginning of the file
            data = fin.read(record_size * int(n_records))
            if not data:
                print("Empty File ",file_path,"🔴🔴")
                return None
            # Unpack the binary data
            data=np.frombuffer(data, dtype=np.dtype(format))
        return data
    except FileNotFoundError:
        print(f"The file '{file_path}' Not Found.")
        
def write_binary_file(file_path,data_to_write,format):
    '''
    data_to_write: array of values with format as passed
    format: format of each element
    '''
    try:
        with open(file_path, "ab") as fout:
            # Pack the entire array into binary data
            binary_data = struct.pack(len(data_to_write)*format, *data_to_write.flatten())
            fout.write(binary_data)
    except FileNotFoundError:
        print(f"The file '{file_path}' could not be created.")

def read_binary_file_chunk(file_path, record_format, start_index, chunk_size=10,dictionary_format=False):
    """
    This Function Reads Chunk from a binary File
    If remaining from file are < chunk size they are returned normally

    file_path:Path of the file to be read from
    record_format: format of the record ex:f"4I" 4 integers
    start_index: index of the record from which we start reading [0_indexed]
    chunk_size: no of records to be retrieved

    @return : None in case out of index of file
              the records
    """

    # Calculate record size
    record_size = struct.calcsize(record_format)

    # Open the binary file for reading
    with open(file_path, "rb") as fin:
        fin.seek(
            start_index * record_size
        )  # Move the file pointer to the calculated offset

        # Read a chunk of records
        # .read() moves the file pointer (cursor) forward by the number of bytes read.
        chunk_data = fin.read(record_size * (chunk_size))
        if len(chunk_data) == 0:
            print("Out Of File Index 🔥🔥")
            return None

        # file_size = os.path.getsize(file_path)
        # print("Current file position:", fin.tell())
        # print("File size:", file_size,"record_format",record_format,"record_size",record_size,"chunk_data len",len(chunk_data))

        if dictionary_format:
            records={}
            for i in range(0, len(chunk_data), record_size):
                #TODO Remove this loop @Basma Elhoseny
                unpacked_record = struct.unpack(record_format, chunk_data[i : i + record_size])
                id, vector = unpacked_record[0], unpacked_record[1:]
                records[id]=np.array(vector)
            return records

        # Unpack Data
        records = []
        for i in range(0, len(chunk_data), record_size):
            unpacked_record = struct.unpack(
                record_format, chunk_data[i : i + record_size]
            )
            id, vector = unpacked_record[0], unpacked_record[1:]
            record = {"id": id, "embed": list(vector)}
            records.append(record)
        return records
def empty_folder(folder_path):
    """
    Function to Empty a folder given its path
    @param folder_path : path of the folder to be deleted
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Created new ", folder_path, "successfully")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error while deleting {file_path}: {e}")
    print("Deleted", folder_path, "successfully")


def extract_embeds(dict):
    # {505: {'id': 505, 'embed': [0.8,....]}} --> [[0.8,....],[.......]]
    return [entry["embed"] for entry in dict.values()]


def extract_embeds_array(arr):
    return np.array([entry["embed"] for entry in arr])


def cal_score(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity


def calculate_offset(record_id: int) -> int:
    # Calculate the offset for a given record ID
    record_size = struct.calcsize("I70f")
    return (record_id) * record_size


def read_multiple_records_by_id(file_path, records_id: List[int],dictionary_format=False):
    record_size = struct.calcsize("I70f")
    records = {}

    records_dictionary={}

    with open(file_path, "rb") as fin:
        for i in range(len(records_id)):
            offset = calculate_offset(records_id[i])
            fin.seek(offset)  # Move the file pointer to the calculated offset
            data = fin.read(record_size)
            if not data:
                records[records_id[i]] = None
                continue

            # Unpack the binary data into a dictionary
            unpacked_data = struct.unpack("I70f", data)
            id_value, floats = unpacked_data[0], unpacked_data[1:]

            if dictionary_format:
                records_dictionary[id_value]=list(floats)
            else:
                # Create and return the record dictionary
                record = {"id": id_value, "embed": list(floats)}
                records[records_id[i]] = record

    if dictionary_format: return records_dictionary
    return records

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

def get_top_k(top_elements,new_array):
    for new_element in new_array:
        if(new_element> min(top_elements, key=lambda x: x[0])):
            arg_min=min(enumerate(top_elements), key=lambda x: x[1][0])[0]
            top_elements[arg_min]=new_element
    return sorted(top_elements, reverse=True)

# print(get_top_k([(-1,None),(-1,None),(10,None),(-1,None)],[(1,1),(5,2),(0,3),(-8,4),(10,5),(800,6),(810,7),(145,8),(36,9),(78,10)]))