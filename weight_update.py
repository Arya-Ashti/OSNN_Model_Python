import numpy as np
from numpy.linalg import LinAlgError

def update_weights(w, batch, pred, mu, alpha, lam, centers, widths, eta):
    '''
    this function will update the weights using the newton raphson update method. It uses another function define below that will
    generate the basis vector for a list of samples

    parameters:
    - w : weights
    - batch : current batch of samples
    - pred : predicted values of samples in batch
    - mu : pseudolabels of samples in batch
    - alpha : L2 regularisation parameter
    - lamb : manifold regularisation parameter
    - centers : 2D array of centers
    - widths : widths of centers
    - eta : learning rate

    returns:
    - updated weights
    '''

    #compute basis vector
    basis_vector = gaussian_basis_vector(batch, centers, widths)

    #gradient of cross entropy
    labelled_grad = np.zeros(w.shape) #initialise running sums
    unlabelled_grad = np.zeros(w.shape) #initialise running sums
    L, U = 0, 0
    
    for i in range(len(batch)):
        if batch[i][-1] == -1: #if it's an unlabelled example, add to the unlabelled running sum
            unlabelled_grad += (pred[i] - mu[i]) * basis_vector[i]
            U += 1
        else: #if it's labelled, then add to the labelled running sum
            labelled_grad +=  (pred[i] - batch[i][-1]) * basis_vector[i]
            L += 1

    reg_term_grad = (alpha/len(batch)) * w #L2 regularisation term
        
    l = 1/L if L > 0 else 0
    u = 1/U if U > 0 else 0
    
    loss_grad = l*labelled_grad + lam*u*unlabelled_grad + reg_term_grad #compute total loss gradient

    #second derivative of cross entropy, i.e hessian
    labelled_hess = np.zeros((len(w), len(w))) #initialise running sums
    unlabelled_hess = np.zeros((len(w), len(w))) #initialise running sums
    
    for i in range(len(batch)):
        outer_prod = np.outer(basis_vector[i],basis_vector[i])
        if batch[i][-1] == -1: #if it's an unlabelled example, add to the unlabelled running sum
            unlabelled_hess += pred[i]*(1-pred[i]) * outer_prod
        else: #if it's labelled, then add to the labelled running sum
            labelled_hess +=  pred[i]*(1-pred[i]) * outer_prod

    #L2 reg term to add to the diagonal of the hessian
    reg_term_hess = (alpha/len(batch)) * np.eye(len(w))
    hessian = l*labelled_hess + u*unlabelled_hess + reg_term_hess

    #inverse of above
    hessian_inv = chol_inv(hessian)

    #gradient of weights
    weight_grad = eta * np.dot(hessian_inv, loss_grad)
    
    updated_weights = w - weight_grad
    return updated_weights
    
'''
def chol_inv(hessian):
    
    #A function to return the inverse of a hessian using Cholesky decomposition
    
        #perform Cholesky decomposition
        L = np.linalg.cholesky(hessian)
    
        #compute the inverse of L
        L_inv = np.linalg.inv(L)
        
        #compute the inverse of the original matrix A
        hessian_inv = L_inv.T @ L_inv
    
        return hessian_inv

'''

def chol_inv(hessian, max_attempts=10):
    '''
    A function to return the inverse of a Hessian using Cholesky decomposition.
    This function will add random noise to the diagonal if the Hessian is not positive definite.

    Parameters:
    - hessian : The Hessian matrix to invert.
    - noise_level : The initial noise level to add to the diagonal.
    - max_attempts : Maximum number of attempts to add noise and invert.
    '''
    #initialise noise level
    noise_level = 1e-4
    for i in range(max_attempts):
        try:
            #try to perform cholesky decomposition
            L = np.linalg.cholesky(hessian)
            
            #compute the inverse of it
            L_inv = np.linalg.inv(L)
            
            #compute the inverse of the original matrix
            hessian_inv = L_inv.T @ L_inv

            if i > 0:
                print(f"Hessian noise adding counter: {i}")
                
            return hessian_inv  #return the inverse of the Hessian
        
        except LinAlgError:
            #if Hessian is not positive-definite, add random noise to the diagonal of the matrix
            hessian += np.eye(hessian.shape[0]) * noise_level*(i+1)
            
    #raise an error if all attempts fail
    raise LinAlgError("Hessian matrix is not positive definite even after adding noise.")
    
def gaussian_basis_vector(samples, centers, widths):    
    '''
    this function is to return a numpy array containing all the basis vectors, inputs a numpy array
    '''

    samples = samples[:,:14]
    
    num_samples = len(samples)
    num_centers = len(centers)
    
    #first initialise a vector to store the basis vector for all samples
    basis_vector = np.zeros((num_samples, num_centers))
    
    for i in range(num_samples):
        #each sample will have H (number of centers) amount of phi's 
        phi_vector = [] #initiialise the phi vector for each sample
        for j in range(num_centers):
            phi = gaussian_basis(samples[i], centers[j], widths[j]) #calc the phi value
            phi_vector.append(phi) #append to the phi vector for that sample
            
        #once we have a vector containing all the phi's for that sample, convert it to a numpy array and append it onto the basis vector matrix
        phi_vector = np.array(phi_vector)
        basis_vector[i] = phi_vector      
    
    return np.array(basis_vector)

def gaussian_basis(sample, center, width):
    '''
    this function evaluates the gaussian basis function
    '''
    numerator = -(np.sum((sample-center)**2))
    denominator = 2*(width**2)

    exponent = numerator / denominator
    exponent = np.clip(exponent, -700, 700)
    
    return np.exp(numerator/denominator)