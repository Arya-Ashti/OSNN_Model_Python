import numpy as np

def accuracy(data):
    '''
    a function to return the accuracy of the model throughout training 
    i.e num_correctly_predicted / total_predicted
    '''
    correctly_labelled = 0
    for _, prediction, truth, _ in data:
        if prediction == truth:
            correctly_labelled += 1

    return correctly_labelled/len(data)

def prequential_accuary(data, fading_factor = 0.99):
    '''
    this function calculates the prequential accuracy
    '''
    #initialise
    weight = 1
    accuracy = 0
    weight_sum = 0
    
    for _, prediction, true_label, _ in data:
        if prediction == true_label: #if predicted correctly
            accuracy += weight #add to the numerator
            
        weight_sum += weight #always add to the denominator

        weight *= fading_factor #update weight for next sample

    preq_acc = 100 * accuracy/weight_sum
    
    return preq_acc

def recall(data):
    '''
    A function to return the recall evaluation of the data
    '''
    TP = 0
    FN = 0
    for _, prediction, true_label, _ in data:
        if true_label == 1:
            if prediction == 1:
                TP += 1
            elif prediction == 0:
                FN += 1

    if TP + FN == 0:
        print("Divide by zero error in Recall calculation: returned 0")
        return 0
        
    eval = (TP)/(TP + FN)
    return eval, TP, FN

def specificity(data):
    '''
    a function to return the specificity evaluation of the data
    '''
    FP = 0
    TN = 0
    for _, prediction, true_label, _ in data:
        if true_label == 0:
            if prediction == 0:
                TN += 1
            elif prediction == 1:
                FP += 1

    if TN + FP == 0:
        print("Divide by zero error in Specificity calculation: returned 0")
        return 0

    eval = TN / (TN + FP)
    return eval, TN, FP

def geometric_mean(recall, specificity):
    '''
    returns the geometric mean between the recall and specificity
    '''
    return np.sqrt(recall * specificity)

def MCC(TP, TN, FP, FN):
    '''
    function to return the Matthew Correlation Coefficient
    '''
    
    accuracy = (TP*TN) - (FP*FN)
    total = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    if total == 0:
        print("divide by zero error in MCC calculation")
        return
    
    return accuracy/total