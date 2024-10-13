import numpy as np

def predict_multiple(samples, centers, widths, weights):
    '''
    a function to return the predictions of a list of samples
    '''
    #remove label columns
    samples = samples[:,:14] 
    predictions = [] #initialise
    
    for sample in samples:
        predictions.append(predict(sample, centers, widths, weights))

    return np.array(predictions)
    

def predict(sample, centers, widths, weights):
    '''
    a function that returns the prediction for a sample
    '''

    #first ensure label column is removed
    sample = sample[:14]
    
    phi_vector = [] # initialise the phi vector (i.e hidden layer values)
    
    for i in range(len(centers)): #for all centers
            phi = gaussian_basis(sample, centers[i], widths[i]) #calculate its phi value
            phi_vector.append(phi) #append

    phi_vector = np.array(phi_vector) #convert to numpy array
    
    z = np.sum(weights * phi_vector)
    
    prediction = 1 / (1 + np.exp(-z))
                                      
    return prediction

def gaussian_basis(sample, center, width):
    '''
    this function evaluates the gaussian basis function
    '''
    numerator = -(np.sum((sample-center)**2))
    denominator = 2*(width**2)
    
    exponent = numerator / denominator
    exponent = np.clip(exponent, -700, 700)
    
    return np.exp(numerator/denominator)