import numpy as np

def update_widths(centers, beta):
    '''
    this function is to update the widths that will be used when calculating the basis vector
    
    parameters:
    - beta : RBF width parameter (user defined)
    - centers: the trained centers of the current batch

    returns:
    - an np.array containing all the widths
    '''
    #the number of widths equate to the number of centers that we have (i.e the number of nodes in the hidden layer)
    num_widths = len(centers)
    
    #initialising an array to store the updated widths
    updated_widths = np.zeros(num_widths) 

    for i in range(num_widths):
        #keep a running sum for each width that we calculate so that we can divide it by the number of centers to get an average.
        sum_dist = 0
        
        for j in range(num_widths):
            if i != j:
                #calc the euclidean distance from center i to all the others and add them up in the running total
                dist = np.linalg.norm(centers[i] - centers[j])
                sum_dist += dist

        #calculuate the updated width using the RBF_width parameter (i.e beta)
        updated_widths[i] = sum_dist * (beta / (num_widths))

    #clip to prevent extreme cases
    updated_widths = np.maximum(updated_widths, 1e-5)
    
    return updated_widths