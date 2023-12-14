# import heapq

# # Assuming cal_score is a function that calculates the similarity score between two vectors
# def cal_score(query, vector):
#     # Your implementation of calculating the similarity score
#     pass

# # Your data structure (records) containing vectors
# records = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]}

# # Example query vector
# query = [0.5, 0.6, 0.7]

# # Number of top elements you want to retrieve
# top_k = 2

# # Calculate the scores and get the top k without sorting the entire list
# scores = []
# for id, vector in records.items():
#     score = cal_score(query, vector)
#     heapq.heappushpop(scores, (score, id))  # Use a min-heap to keep track of top k scores

# # The 'scores' list now contains the top k elements
# print("Top", top_k, "elements:", scores)
import numpy as np
def get_top_k(top_elements,new_array):
    for new_element in new_array:
        if(new_element> min(top_elements, key=lambda x: x[0])):
            arg_min=min(enumerate(top_elements), key=lambda x: x[1][0])[0]
            top_elements[arg_min]=new_element
    return sorted(top_elements, reverse=True)

print(get_top_k([(-1,None),(-1,None),(10,None),(-1,None)],[(1,1),(5,2),(0,3),(-8,4),(10,5),(800,6),(810,7),(145,8),(36,9),(78,10)]))