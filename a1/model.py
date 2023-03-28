

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
        
        l = - 1 / D * sum([np.dot(Y[i, :], np.log(P[:, i])) for i in range(D)])
        r = lambd * np.sum(self.W**2)
        return  l + r
    
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
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        
        Returns
        -------
        W_grads : gradients for weight martix (W)
        b_grads : gradients for bias matrix (b)
        """
        D = len(X)
        
        P = self.evaluate(X)
        g = -(Y.T - P)
        
        W_grads = 1 / D * np.matmul(g, X) + 2 * lambd * self.W
        b_grads = 1 / D * np.sum(g, axis=1)
        
        return W_grads, np.expand_dims(b_grads, axis=1)

    def computeGradsNumerical(self, X, Y, lambd):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        
        Returns
        -------
        W_gradsNum : numerically calculated gradients for weight martix (W)
        b_gradsNum : numerically calculated gradients for bias matrix (b)
        """
        W_0, b_0 = self.W, self.b
        eps = 1e-5
        
        # calculate numerical gradients for W
        W_perturb = np.zeros(W_0.shape)
        W_gradsNum = np.zeros(W_0.shape)
        for i in range(self.K):
            for j in range(self.d):
                W_perturb[i, j] = eps
                
                W_tmp = W_0 - W_perturb
                self.W = W_tmp
                cost = self.computeCost(X, Y, lambd)
                
                W_tmp = W_0 + W_perturb
                self.W = W_tmp
                lossDiff = (self.computeCost(X, Y, lambd) - cost) / (2 * eps)
                
                W_gradsNum[i, j] = lossDiff
                W_perturb[i, j] = 0
            
        self.W = W_0
        
        # calculate numerical gradients for b
        b_perturb = np.zeros(b_0.shape)
        b_gradsNum = np.zeros(b_0.shape)
        for i in range(self.K):
            b_perturb[i] = eps
            
            b_tmp = b_0 - b_perturb
            self.b = b_tmp
            cost = self.computeCost(X, Y, lambd)
            
            b_tmp = b_0 + b_perturb
            self.b = b_tmp
            lossDiff = (self.computeCost(X, Y, lambd) - cost) / (2 * eps)
            
            b_gradsNum[i] = lossDiff
            b_perturb[i] = 0
            
        self.b = b_0
        
        return W_gradsNum, np.squeeze(b_gradsNum)
    
    def train(self, X, Y, lambd, eta):
        W_grads, b_grads = self.computeGrads(X, Y, lambd)
        
        self.W = self.W - eta * W_grads
        self.b = self.b - eta * b_grads
        
