

import numpy as np
from misc import softMax

class linearClassifier:
    def __init__(self, K, d):
        self.K = K
        self.d = d
        
        # init weights
        self.W = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, d)
        )
        
        # init bias
        self.b = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, 1)
        )

    def evaluate(self, X):
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        S = np.matmul(self.W, X.T) + self.b
        return softMax(S)
    
    def computeCost(self, X, Y, lambd):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd : regularization parameter
        
        Returns
        -------
        J : cross-entropy loss w. L2-regularization
        """
        D = len(X)
        P = self.evaluate(X)
        
        loss = [-np.dot(Y[i, :], np.log(P[:, i])) for i in range(D)]
        J    = 1 / D * sum(loss) + lambd * np.sum(self.W**2)
        return J
    
    def computeAcc(self, X, k):
        """
        Parameters
        ----------
        X : Nxd data matrix
        k : Nx1 ground-truth label vector
        Returns
        -------
        acc : accuracy score
        """
        P     = self.evaluate(X)
        preds = np.argmax(P, axis=0)

        return np.mean([preds == k])
    
    def computeGrads(self, X, Y, lambd):
        J = self.computeCost(X, Y, lambd)
        
        
        return 0

