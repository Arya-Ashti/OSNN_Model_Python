import numpy as np

def cross_entropy_loss(batch, pred, mu, w, alpha, lam):
    '''
    This function returns the cross entropy loss of the network

    parameters:
    - batch : current batch, 15 columns
    - pred : predicted value f
    - mu : pseudolabels
    - w : weights
    - alpha : L2 regularisation term

    returns:
    - loss
    '''

    N = len(batch) #chunk size

    #set a value for epsilon to clip predicted values and clip them, this is to avoid computing log(0)
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)

    #initialise contribution to loss of of each class and unlabelled.
    class0_loss = 0
    class1_loss = 0
    unlabelled_loss = 0
    L = 0
    U = 0

    for i in range(N): #iterate over all samples in batch
        
        if batch[i][-1] == 1: #if it's labelled and of class 1, add to the class 1 running sum
            class1_loss += np.log(pred[i])
            L += 1
            
        elif batch[i][-1] == 0: #if it's labelled and of class 0, add to the class 0 running sum
            class0_loss += np.log(1-pred[i])
            L += 1

        else: #otherwise, it's unlabelled:
            unlabelled_loss += mu[i]*np.log(pred[i]) + (1 - mu[i])*np.log(1-pred[i])
            U += 1

    #to avoid dividing by 0 errors:
    l = 0 if L == 0 else 1/L
    u = 0 if U == 0 else 1/U
    
    #l_2 regularisation loss
    l2_reg = (alpha / (2*N)) * (np.linalg.norm(w)**2)

    loss = -l*(class1_loss + class0_loss) - (lam*u)*(unlabelled_loss) + l2_reg
    
    return loss