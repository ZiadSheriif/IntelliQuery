
import heapq

# Example list of tuples (vector, priority score)
scores = [([1, 2, 3], 85), ([4, 5, 6], 92), ([7, 8, 9], 78), ([10, 11, 12], 90)]
top_k = 5

# Use heapq.nlargest with a custom key function
top_k_elements = heapq.nlargest(top_k, scores, key=lambda x: x[1])

# Print the result
print("Top", top_k, "elements:", top_k_elements)