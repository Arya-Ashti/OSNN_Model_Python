import pandas as pd
import numpy as np

def import_data(filename, delim = ";"):
    '''
    this function will load the data in and remove the unix timestamp
    filename should be a .csv file, i.e input as import_data(tomcat.csv, ";")
    '''
    
    #load it into the dataframe
    df = pd.read_csv(filename, delimiter = delim)

    #convert it to a numpy array
    data = df.to_numpy()

    #remove the first column, i.e unix timestamp column
    dataset = data[:, 1:]

    return dataset

def import_data_with_unixtimestamp(filename, delim = ";"):
    '''
    this function will load the data in and remove the unix timestamp
    filename should be a .csv file, i.e input as import_data(tomcat.csv, ";")
    '''
    
    #load it into the dataframe
    df = pd.read_csv(filename, delimiter = delim)

    #convert it to a numpy array
    data = df.to_numpy()

    return data
    
def import_predictions(filename, delim = ";"):
    '''
    this function will load the data in and remove the unix timestamp
    filename should be a .csv file, i.e input as import_data(tomcat.csv, ";")
    '''
    
    #load it into the dataframe
    df = pd.read_csv(filename, delimiter = delim)

    #convert it to a numpy array
    data = df.to_numpy()

    return data

def save_data(filename, array, delimiter=';'):

    #covert np array to dataframe
    df = pd.DataFrame(array)
    
    #save dataframe to .csv
    df.to_csv(filename, sep=delimiter, index=False)

    return

def save_predictions(filename, array, delimiter=';'):

    #covert np array to dataframe
    df = pd.DataFrame(array, columns=['predicted probability','prediction', 'true_label', 'contains_bug'])
    
    #save dataframe to .csv
    df.to_csv(filename, sep=delimiter, index=False)

    return