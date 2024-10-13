import numpy as np

def pseudolabels_calc(C, batch, pred, c_pred, alpha, gamma):
    '''
    this function is to calculate the pseudolabels which will be used for updating the weights and calculating the loss
    we will require the labels of the labelled data, and the predicted labels of the unlabelled data.
    in order to calculate the predicted values, we will require to run the unlabelled data through the network

    parameters: 
    - C : centers
    - batch : unlabelled samples
    - pred : predicted values for all terms in the batch
    - c_pred : predicted values of the centers
    - alpha : L2 regularisation term
    - gamma : RBFN width
    '''
    
    #first we add a column to the end of the centers
    C_with_labels = np.hstack((C, -1 * np.ones((len(C), 1))))

    #concatinate the predicted values of the batch and centers
    predicted = np.append(c_pred, pred)

    #combine the centers and the batch
    V = np.vstack((C_with_labels, batch))
    num_vertices = len(V)

    # initialise
    pseudolabels = []

    #calculate the similarity matrix
    S_matrix = similarity_matrix(V, gamma)

    for i in range(num_vertices):
        denominator = 0
        numerator = 0
        
        for j in range(num_vertices):
            denominator += S_matrix[i][j]
            if V[j][-1] == -1:
                numerator += S_matrix[i][j] * predicted[j] #if unlabelled, use predicted value
            else:
                numerator += S_matrix[i][j] * V[j][-1] #if labelled, use label

        #append the pseudolabel for the i'th term to our array
        pseudolabels.append(numerator/denominator)

    #convert to np.array
    pseudolabels = np.array(pseudolabels)

    #remove pseudolabels corresponding to centers
    pseudolabels = pseudolabels[len(C):]
    
    return pseudolabels

def sigma_calc(V, gamma):
    '''
    a function to calculate the distance from each vertex to it's nearest neighbour

    parameters:
    - V : 2D np array of the vertices
    - gamma : RBFN width
    '''
    num_samples = len(V)
    min_distances = np.full(num_samples, np.inf) # initialise with each minimum being infinity initially
    
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                #calc the distance from the current vertex to the j^th one
                dist = np.linalg.norm(V[i] - V[j])
                #if it's less than the current minimum, replace it
                if dist < min_distances[i]:
                    min_distances[i] = dist

    sigmas = min_distances * gamma

    #clip them
    sigmas = np.maximum(sigmas, 1e-10)
    
    return sigmas

def similarity_matrix(V, gamma):
    '''
    a function to generate the similarity matrix of a dataset V and some provided gamma

    parameters:
    - V : 2D np array of the vertices
    - gamma : RBFN width
    '''
    sigmas = sigma_calc(V, gamma)
    num_samples = len(V)

    
    similarity_matrix = np.zeros((num_samples, num_samples))
    
    for i in range(num_samples):
        for j in range(num_samples):
            similarity_matrix[i][j] = np.exp(-(np.linalg.norm(V[i]-V[j])**2) / (2*sigmas[i]**2))
            
    return similarity_matrix