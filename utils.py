import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    # Sample data: 10 vectors with 70 features each
    data = np.random.uniform(-1, 1, size=(k, 70))

    # Write data to a text file
    file_path = "random_data.txt"
    np.savetxt(file_path, data)
    read_data = np.loadtxt(file_path)
    # print(read_data)

generate_random(1000)