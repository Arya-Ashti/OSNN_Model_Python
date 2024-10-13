import numpy as np

def train_centers(centers, batch):
    '''
    this function trains the centers using old centers and the current batch

    parameters:
    - centers : the old centers to be trained, 14 columns
    - batch: the current batch, 15 columns

    outputs:
    - the trained centers, 14 columns
    '''
    
    num_centers = len(centers)

    #create an empty array to store the trained centers, the number of columns equal to the length of each center
    trained_centers = np.empty((0, len(centers[0])))

    #calculate the regions of each center:
    regions = region_split(centers, batch)

    #train centers
    trained_centers = np.array([train_center(centers[i], regions[i]) for i in range(num_centers)])
        
    return trained_centers

def train_center(center, region):
    '''
    This function will train a particular center based on the region of its closest examples

    parameters:
    - center : the center we want to train, expects np.array
    - region : the region of samples which have this center as the nearest one, expects np.array

    returns:
    - trained_center
    '''
    
    #split samples into labelled and unlabelled batches
    labelled = np.array([sample for sample in region if sample[-1] in [0,1]])
    unlabelled = np.array([sample for sample in region if sample[-1] == -1])

    #split the labelled batch into majority and minority sets
    maj, min = split_labelled_batch(labelled)

    #remove the last column of the majority, minority, and unlabelled
    maj = maj[:,:14] if len(maj) > 0 else maj
    min = min[:,:14] if len(min) > 0 else min
    unlabelled = unlabelled[:,:14] if len(unlabelled) != 0 else unlabelled
    
    #handle cases if there are no samples in the region
    L = 1/len(labelled) if len(labelled) > 0 else 0
    U = 1/len(unlabelled) if len(unlabelled) > 0 else 0

    #since centers are only impacted by samples in their region, if their region is empty, then don't update the center
    if L==0 and U==0:
        return center

    #calculated the updated center
    maj_sum = np.sum(maj, axis=0) if len(maj) > 0 else 0 #sum of elements in majority set, 0 if empty
    min_sum = np.sum(min, axis=0) if len(min) > 0 else 0 #sum of elements in minority set, 0 if empty
    unlabelled_sum = np.sum(unlabelled, axis=0) if U > 0 else 0 #sum of elements in unlabelled set, 0 if empty

    numerator = (L * (maj_sum - min_sum)) + (U * unlabelled_sum)
    denominator = (L * (len(maj) - len(min))) + (U * len(unlabelled))

    if denominator == 0:
        return center

    return numerator / denominator
    
def region_split(centers, batch):
    '''
    this function splits the samples into regions based on which center they're closest to
    initialise a list containing empty lists to store the samples for each region

    parameters:
    - centers : np.array of centers (14 columns)
    - batch : np.array of batch (15 columns)

    returns:
    - regions : 15 columns
    '''
    
    #initilise the regions
    regions = [[] for _ in centers]
    
    for i in range(len(batch)): #iterate over all samples
        min_dist = np.inf #define the initial minimum distance as infininty
        
        for j in range(len(centers)): #compare sample to all centers
            dist = euclidean_distance(batch[i, 0:14], centers[j]) #calculate the distance between the sample and each center
            
            if dist < min_dist: #if that center is the new minimum center:
                min_dist = dist #store the new minimum distance
                closest_centroid = j #keep track of which centroid this sample was closest too
                
        regions[closest_centroid].append(batch[i])
  
    regions = [np.array(region) for region in regions]
    
    return regions

def euclidean_distance(x1,x2):
    '''
    this function returns the euclidean distance between vectors x1 and x2
    '''
    return np.linalg.norm(x1-x2)
    

def split_labelled_batch(labelled_batch):
    '''
    this function separates the samples in the labelled batch into majority and minority sets
    if there are no labelled samples, it should return two empty sets

    parameters:
    - labelled_batch: set of labelled samples, (14 columns, i.e removal of the label columns)
    '''
    L = len(labelled_batch)

    #initialise the sets
    minority = np.empty((0,14)) 
    majority = np.empty((0,14))
    
    #return 2 empty sets if there's no labelled samples
    if L == 0:
        return majority, minority 

    majority = np.array([sample for sample in labelled_batch if sample[-1]==0])
    minority = np.array([sample for sample in labelled_batch if sample[-1]==1])

    #ensure minority has fewer samples
    if len(minority) > len(majority):
        minority, majority = majority, minority
        
    return majority, minority