import os.path
import struct
import numpy as np

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def uint2int(arr):
    if len(arr.shape) == 1:  # Lista
        return np.array([int(a) for a in arr.flat])
    else:  # Tablica
        return np.array([int(a) for a in arr.flat]).reshape(len(arr), len(arr[0]))


def getMNISTdata(filePath):

    with open(filePath, 'rb') as f:
        if "images" in filePath:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
            data = data.reshape((size, nrows * ncols))
            
        elif "labels" in filePath:
            magic, size = struct.unpack(">II", f.read(8))
            nrows, ncols = struct.unpack(">II", f.read(8))
            data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        else:
            data = ""
    return data

def labelsToNumbers(labs):
    length = len(labs)
    outputLab = np.zeros(length, dtype=int)

    labList = np.array(['mulberry leaf', 'fruitcitere', 'pomme jacquot', 'croton', 'caricature plant', 'papaya', 'jackfruit', 'sweet olive', 'hibiscus', 'ficus', 'lychee', 'vieux garcon', 'geranium', 'sweet potato', 'rose', 'thevetia', 'ashanti blood', 'beaumier du perou', 'ketembilla', 'coeur demoiselle', 'eggplant', 'betel', 'coffee', 'chrysanthemum', 'chocolate tree', 'duranta gold', 'chinese guava', 'pimento', 'bitter orange', 'guava', 'star apple', 'barbados cherry'])
    for i in range(length):
        for j in range(len(labList)):
            if labs[i] == labList[j]:
                outputLab[i] = j
    
    return outputLab