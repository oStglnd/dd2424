
import numpy as np
import pickle

def softMax(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    P : kxN probability matrix w. applied softmax activation
    """
    S = np.exp(S)
    return S / np.sum(S, axis=0)

def sigmoid(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    P : kxN probability matrix w. sigmoid activations
    """
    return 1 / (1 + np.exp(-S))

def multBCE(p: np.array, y: np.array, K: int) -> np.array:
    """
    Parameters
    ----------
    P : kx1 probability vector
    Y : 1xk one-hot encoded vectro
    
    Returns
    -------
    l : multiple binary cross-entropy loss f. (p(x), y)
    """
    ones = np.ones(K)
    l = - 1 / K * (np.dot(ones - y, np.log(ones - p)) + np.dot(y, np.log(ones - p)))
    return l

def oneHotEncode(k: np.array) -> np.array:
    """
    Parameters
    ----------
    k : Nx1 label vector

    Returns
    -------
    Y: NxK one-hot encoded label matrix
    """
    return np.array([[
        1 if idx == label else 0 for idx in range(10)]
         for label in k]
    )

def getCifar(
        fpath: str, 
        fname: str or list
    ) -> (np.array, np.array, np.array):
    """
    Parameters
    ----------
    fpath : str
    
    Returns
    -------
    X: Nxd data matrix
    k: Nx1 label vector
    Y: NxK one-hot encoded matrix
    """
    # open batch w. pickle
    with open(fpath + fname, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        
    # extract data and convert to numPy arrays
    X    = np.array(batch[b'data'])
    k    = np.array(batch[b'labels'])
    Y    = oneHotEncode(k)
        
    # delete batch from memory
    del batch
    
    return X, k, Y
        
def imgFlip(X: np.array, prob: float) -> np.array:
    """
    Parameters
    ----------
    X : nxd flattened img. array
    angle : int

    Returns
    -------
    X : nxd shuffled, flattened img. array w. some flipped inputs
    """
    # get shape
    n, d = X.shape
    
    # get sampls along idx axis
    # and convert to boolean array
    idxs = np.random.rand(n) < prob
    
    # split data
    X_flipped = X[idxs].copy()
    N = len(X_flipped)
    
    # flip selected data
    X_flipped = X_flipped.reshape((N, 3, 32, 32))
    X_flipped = np.flip(X_flipped, axis=3).reshape((N, d))
    
    # X_flipped = X_flipped.transpose(0, 2, 3, 1)
    # X_flipped = np.flip(X_flipped, axis=2)
    # X_flipped = X_flipped.transpose(0, 3, 1, 2).reshape((N, d))
    
    # concatenate back into one array
    X[idxs] = X_flipped
    
    return X