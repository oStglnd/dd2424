

import numpy as np
from misc import softMax

class neuralNetwork:
    def __init__(
            self, 
            K: int, 
            d: int,
            m: int,
            seed: int
        ):
        
        # init weight dims
        self.K = K
        self.d = d
        self.m = m
        
        # set seed
        np.random.seed(seed)
        
        # init weights
        self.weights = {
            'W1': np.random.normal(
                    loc=0, 
                    scale=1/np.sqrt(self.d), 
                    size=(self.m, self.d)
                ),
            'W2': np.random.normal(
                    loc=0, 
                    scale=1/np.sqrt(self.m), 
                    size=(self.K, self.m)
                ),
            'b1': np.random.normal(
                    loc=0, 
                    scale=0.0, 
                    size=(self.m, 1)
                ), 
            'b2': np.random.normal(
                loc=0, 
                scale=0.0, 
                size=(self.K, 1)
                )
        }
        
        # # init weights, layer 1
        # self.W1 = np.random.normal(
        #     loc=0, 
        #     scale=1/np.sqrt(self.d), 
        #     size=(self.m, self.d)
        # )
        
        # # init bias, layer 1
        # self.b1 = np.random.normal(
        #     loc=0, 
        #     scale=0.0, 
        #     size=(self.m, 1)
        # )
        
        # # init weights, layer 2
        # self.W2 = np.random.normal(
        #     loc=0, 
        #     scale=1/np.sqrt(self.m), 
        #     size=(self.K, self.m)
        # )
        
        # # init bias, layer 1
        # self.b2 = np.random.normal(
        #     loc=0, 
        #     scale=0.0, 
        #     size=(self.K, 1)
        # )

    def evaluate(
            self, 
            X: np.array,
            train: bool
        ) -> np.array:
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        P : KxN score matrix w. softmax activation
        """
        s1 = self.weights['W1'] @ X.T + self.weights['b1'] # m x N
        h = np.maximum(0, s1)        # m x N
        s = self.weights['W2'] @ h + self.weights['b2']    # K x N
        P = softMax(s)               # K x N
        
        if not train:
            return P
        else:
            return P, h, s1
    
    def predict(
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
        P = self.evaluate(X, train=False)
        preds = np.argmax(
            P, 
            axis=0
        )
        
        return preds
    
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
        D = len(X)
        
        # get probabilities
        P = self.evaluate(X, train=False)
        
        # evaluate loss and regularization term
        l = - D**-1 * np.sum(Y.T * np.log(P))
        r = lambd * (np.sum(np.square(self.weights['W1'])) + np.sum(np.square(self.weights['W2'])))
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
        preds = self.predict(X, train=False)
        return np.mean(preds == k)
    
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
        P, h, s1 = self.evaluate(X, train=True)
        g = -(Y.T - P) # K x N
        
        # evaluate gradients for second layer
        W2_grads = D**-1 * g @ h.T + 2 * lambd * self.weights['W2'] # K x m
        b2_grads = D**-1 * np.sum(g, axis=1) # K x 1
        
        # evaluate gradients for first layer
        g = self.weights['W2'].T @ g # m x N
        
        # get indicator for prev layer
        idx = h > 0
        h[idx], h[~idx] = 1, 0  # m x N
        
        # get new g
        g = g * h # m x N
        
        #@ np.diag(np.maximum(0, s1))
        W1_grads = D**-1 * g @ X + 2 * lambd * self.weights['W1'] # m x D
        b1_grads = D**-1 * np.sum(g, axis=1) # m x 1
        
        # expand bias vectors to account for proper shape
        b1_grads = np.expand_dims(b1_grads, axis=1)
        b2_grads = np.expand_dims(b2_grads, axis=1)
        
        return W1_grads, W2_grads, b1_grads, b2_grads

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
        weights = ['W1', 'W2', 'b1', 'b2']
        gradsList = []
        
        for weight in weights:
            shape = self.weights[weight].shape
            w_perturb = np.zeros(shape)
            w_gradsNum = np.zeros(shape)
            w_0 = self.weights[weight].copy()
            
            for i in range(self.K):
                for j in range(min(shape[1], 100)):
            
                    # add perturbation
                    w_perturb[i, j] = eps
                    
                    # perturb weight vector negatively
                    # and compute cost
                    w_tmp = w_0 - w_perturb
                    self.weights[weight] = w_tmp
                    _, cost1 = self.computeCost(X, Y, lambd)
                
                    # perturb weight vector positively
                    # and compute cost
                    w_tmp = w_0 + w_perturb
                    self.weights[weight] = w_tmp
                    _, cost2 = self.computeCost(X, Y, lambd)
                    lossDiff = (cost2 - cost1) / (2 * eps)
                    
                    # get numerical grad f. W[i, j]
                    w_gradsNum[i, j] = lossDiff
                    w_perturb[i, j] = 0
        
            # reset weigth vector
            self.weights[weight] = w_0
            gradsList.append(w_gradsNum)
        
        return gradsList
    
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
        grads = self.computeGrads(X, Y, lambd)
        weights = ['W1', 'W2', 'b1', 'b2']
        
        for grad, weight in zip(grads, weights):
            self.weights[weight] -= eta * grad
        
        