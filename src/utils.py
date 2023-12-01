import numpy as np
import shutil
import os

# from sklearn.metrics.pairwise import cosine_similarity

# def get_top_k_similar(target_vector,data,k=10):
#     '''
#     Data: Data Set
#     target_vector:Query
#     '''
#     similarities = cosine_similarity([target_vector], data)
#     most_similar_indices = np.argpartition(-similarities, k, axis=1)[:, :k]
#     k_most_similar_vectors = data[most_similar_indices]
#     return most_similar_indices,k_most_similar_vectors





def generate_random(k=100):
    # Sample data: k vectors with 70 features each
    data = np.random.uniform(-1, 1, size=(k, 70))

    # Write data to a text file
    file_path = "../DataBase/random_data_"+str(k)+".txt"
    np.savetxt(file_path, data)

    # Read Data from File
    # read_data = np.loadtxt(file_path)



def array_to_dictionary(values,keys=None):
    '''
    values: [array of values]
    Keys: [array of Keys] optional if not passed the keys are indexed 0-N
    '''
    if(keys is None):
        keys=range(0,len(values))

    if(len(values)!=len(keys)):
        print ("array_to_dictionary(): InCorrect Size of keys and values")
        return  None
    
    dictionary_data = dict(zip(keys, values))
    return dictionary_data


def get_vector_from_id(data_path,id):
    '''
    function to get the vector by its id [BADDDDDDDD Use Seek]

    '''
    read_data = np.loadtxt(data_path)
    return read_data[id]




def check_dir(path):
   if os.path.exists(path):
    shutil.rmtree(path, ignore_errors=True, onerror=lambda func, path, exc: None)
    os.makedirs(path)


# Test generate_random()
generate_random(10000)