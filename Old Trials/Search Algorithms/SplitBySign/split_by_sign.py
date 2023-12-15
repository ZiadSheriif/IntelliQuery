import numpy as np

# Improvements
# Idea of Tree instead of intersection of lists
# Idea of LSH instead of the list of the regions


def generate_data(size,dim):
    # Sample data: 10 vectors with 70 features each
    data = np.random.uniform(-1, 1, size=(size, dim))
    return data


def generate_query(dim):
# Select the target vector you want to find the k most similar to
    return  np.random.uniform(-1, 1, size=(dim,))



def sorted_list_intersection(list1, list2):
    intersection = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            i += 1
        elif list1[i] > list2[j]:
            j += 1
        else:
            intersection.append(list1[i])
            i += 1
            j += 1

    return intersection


def split_on_sign(data:[[float]],split_on)->int:
    '''
    @param data: Data to categorize
    @split_on: The max count to split on

    @return dictionary of the data splitted by sign +ve and -ve
    '''
    if(split_on is None or split_on>np.shape(data)[1]):
        #split on the whole size
        # split_on=np.shape(data)[1]
        split_on=10

    regions = []
    for col in data[:,:split_on].T:  # Transpose the matrix to iterate over columns
        positive_region = (col >= 0)
        negative_region = (col < 0)
        regions.append(np.where(positive_region)[0])
        regions.append(np.where(negative_region)[0])
    return regions

def search_on_sign(q:[float],regions:[[int]]):
    #  O(m * n), where m is the average length of the input lists, and n is the number of input lists. 
    # Check on sign of the feature
    intersect=None
    split_on=np.shape(regions)[0]//2
    for ind,feature in enumerate(q[:split_on]):
        if(ind==0):
            intersect=regions[0] if feature>=0  else regions[1]
            continue
        if(feature>=0):
            # positive
            intersect=sorted_list_intersection(intersect, regions[2*ind])
        else:
            #negative
            intersect=sorted_list_intersection(intersect, regions[2*ind+1])
    return intersect