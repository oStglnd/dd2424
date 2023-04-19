

import numpy as np
from misc import softMax

class linearClassifier:
    def __init__(
            self, 
            K: int, 
            d: int,
            seed: int
        ):
        
        # init weight dims
        self.K = K
        self.d = d
        
        # set seed
        np.random.seed(seed)
        
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

    def evaluate(
            self, 
            X: np.array
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        # calculate scores
        S = self.W @ X.T + self.b
        
        # return scores w. softmax activation
        return softMax(S)
    
    def computeCost(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float
        ) -> float:
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd : regularization parameter
        
        Returns
        -------
        l : cross entropy loss
        J : cost, i.e. cross-entropy loss w. L2-regularization
        """
        # get size of batch
        D = len(X)
        
        # get probabilities
        P = self.evaluate(X)
        
        # evaluate loss and regularization term
        l = - D**-1 * np.sum(Y.T * np.log(P))
        r = lambd * np.sum(np.square(self.W))
        return  l, l + r
    
    def computeAcc(
            self, 
            X: np.array, 
            k: np.array
        ) -> float:
        """
        Parameters
        ----------
        X : Nxd data matrix
        k : Nx1 ground-truth label vector
        Returns
        -------
        acc : accuracy score
        """
        # get probabilities and predictions
        P     = self.evaluate(X)
        preds = np.argmax(P, axis=0)
        
        # return accuracy
        return np.mean([preds == k])
    
    def computeGrads(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float
        ) -> (np.array, np.array):
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
        # get size of batch
        D = len(X)
        
        # evaluate probabilities and calculate g
        P = self.evaluate(X)
        g = -(Y.T - P)
        
        # get weight gradients and bias gradients
        W_grads = D**-1 * g @ X
        b_grads = D**-1 * np.sum(g, axis=1)
        
        # if regularization
        if lambd > 0:
            W_grads += 2 * lambd * self.W
        
        return W_grads, np.expand_dims(b_grads, axis=1)

    def computeGradsNumerical(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float,
            eps: float
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eps: epsilon for incremental derivative calc.
        
        Returns
        -------
        W_gradsNum : numerically calculated gradients for weight martix (W)
        b_gradsNum : numerically calculated gradients for bias matrix (b)
        """
        # save initial weights
        W_0, b_0 = self.W, self.b
        
        # calculate numerical gradients for W
        W_perturb = np.zeros(self.W.shape)
        W_gradsNum = np.zeros(self.W.shape)
        for i in range(self.K):
            for j in range(self.d):
                W_perturb[i, j] = eps
                
                # perturb weight vector negatively
                # and compute cost
                W_tmp = W_0 - W_perturb
                self.W = W_tmp
                _, cost1 = self.computeCost(X, Y, lambd)
                
                # perturb weight vector positively
                # and compute cost
                W_tmp = W_0 + W_perturb
                self.W = W_tmp
                _, cost2 = self.computeCost(X, Y, lambd)
                lossDiff = (cost2 - cost1) / (2 * eps)
                
                # get numerical grad f. W[i, j]
                W_gradsNum[i, j] = lossDiff
                W_perturb[i, j] = 0
        
        # reset weigth vector
        self.W = W_0
        
        # calculate numerical gradients for b
        b_perturb = np.zeros(b_0.shape)
        b_gradsNum = np.zeros(b_0.shape)
        for i in range(self.K):
            b_perturb[i] = eps
            
            # perturb bias vector negatively
            # and compute cost
            b_tmp = b_0 - b_perturb
            self.b = b_tmp
            _, cost1 = self.computeCost(X, Y, lambd)
            
            # perturb weight vector positively
            # and compute cost
            b_tmp = b_0 + b_perturb
            self.b = b_tmp
            _, cost2 = self.computeCost(X, Y, lambd)
            lossDiff = (cost2 - cost1) / (2 * eps)
            
            # get numerical grad f. b[i]
            b_gradsNum[i] = lossDiff
            b_perturb[i] = 0
            
        # reset bias vector
        self.b = b_0
        
        return W_gradsNum, b_gradsNum
    
    def train(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float, 
            eta: float
        ):
        """
        Parameters
        ----------
        X : Nxd data matrix
        Y : NxK one-hot encoded label matrix
        lambd: regularization parameter
        eta: learning rate
        """
        # get grads from self.computeGrads and update weights
        # w. GD and learning parameter eta
        W_grads, b_grads = self.computeGrads(X, Y, lambd)
        self.W -= eta * W_grads
        self.b -= eta * b_grads
        
