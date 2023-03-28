
import numpy as np
import pickle

def softMax(S):
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    S : dxN score matrix w. applied softmax activation
    """
    S = np.exp(S)
    return S / np.sum(S, axis=0)

def oneHotEncode(k):
    return np.array([[
        1 if idx == label else 0 for idx in range(10)]
         for label in k]
    )

def getCifar(fpath):
    with open(fpath, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        
        # data_dict[file] = np.reshape(
        #     a=batch[b'data'],
        #     newshape=(10000,32,32,3)
        # )
    X    = np.array(batch[b'data'])
    k    = np.array(batch[b'labels'])
    Y    = oneHotEncode(k)
        
    del batch
    
    return X, k, Y
        