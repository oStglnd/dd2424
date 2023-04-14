
import numpy as np
import pickle
import scipy.io as sio

def softMax(S: np.array) -> np.array:
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

def saveAsMat(data, name="model"):
    """ Used to transfer a python model to matlab """
    sio.savemat(f'{name}.mat', {"name": "b"})

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
    
def cyclicLearningRate(
        etaMin: float,
        etaMax: float,
        stepSize: float,
        timeStep: int
    ) -> float:
    l = timeStep // (2 * stepSize)
    
    if (2 * l * stepSize <= timeStep <= (2 * l + 1) * stepSize):
        eta = etaMin + (timeStep - 2 * l * stepSize) / stepSize * (etaMax - etaMin)
    else:
        eta = etaMax - (timeStep - (2 * l + 1) * stepSize) / stepSize * (etaMax - etaMin)
    
    return eta

def imgFlip(X: np.array, prob: float) -> np.array:
    """
    Parameters
    ----------
    X : nxd flattened img. array
    angle : int

    Returns
    -------
    X : nxd flattened img. array w. some flipped inputs
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

    # concatenate back into one array
    X[idxs] = X_flipped
    
    # delete flipped imgs
    del X_flipped
    
    return X

def imgTransl(X: np.array, prob: float) -> np.array:
    """
    Parameters
    ----------
    X : nxd flattened img. array
    angle : int

    Returns
    -------
    X : nxd flattened img. array w. some translated inputs
    """
    # get shape
    n, d = X.shape
    
    # get sampls along idx axis
    # and convert to boolean array
    idxs = np.random.rand(n) < prob
    
    # split data
    X_translated = X[idxs].copy()
    N = len(X_translated)
    
    # translate selected data
    X_translated = X_translated.reshape((N, 3, 32, 32))

    # randomize rollIdx and roll img
    rollIdxV = np.random.choice(np.arange(-3, 3))
    rollIdxH = np.random.choice(np.arange(-3, 3))
    X_translated = np.roll(
        X_translated, 
        (rollIdxV, rollIdxH),
        axis=[2, 3]
    )
    
    # blacken removed pixels
    if (rollIdxV >= 0) and (rollIdxH >= 0):
        X_translated[:, :, :rollIdxH, -rollIdxV:] = 0
    elif (rollIdxV >= 0) and (rollIdxH < 0):
        X_translated[:, :, :rollIdxH, :rollIdxV] = 0
    elif (rollIdxV < 0) and (rollIdxH >= 0):
        X_translated[:, :, -rollIdxH:, -rollIdxV:] = 0
    else:
        X_translated[:, :, -rollIdxH:, :rollIdxV] = 0

    # concatenate back into one array
    X[idxs] = X_translated.reshape((N, d))
    
    # delete flipped imgs
    del X_translated
    
    return X