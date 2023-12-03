import numpy as np
import shutil
import os



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

def extract_embeds(dict):
    # {505: {'id': 505, 'embed': [0.8,....]}} --> [[0.8,....],[.......]]
    return [entry['embed'] for entry in dict.values()]

def extract_embeds_array(arr):
    return np.array([entry['embed'] for entry in arr])

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