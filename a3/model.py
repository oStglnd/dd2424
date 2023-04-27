

import numpy as np
from misc import softMax

class neuralNetwork:
    def __init__(
            self, 
            K: int, 
            d: int,
            m: list,
            alpha: float,
            seed: int
        ):
        
        # init weight dims
        self.K = K
        self.d = d
        
        # init weight dims list
        weightList = [d] + m + [K]
        self.layers = []    
        
        # init batchNorm param
        self.alpha = alpha
        
        # iterate over weight dims
        np.random.seed(seed)
        for m1, m2 in zip(weightList[:-1], weightList[1:]):
            layer = {}
            layer['W'] = np.random.normal(
                    loc=0, 
                    scale=1/np.sqrt(m1), 
                    size=(m2, m1)
            )
            
            for param in ['b', 'mu']:
                layer[param] = np.random.normal(
                    loc=0, 
                    scale=0.0, 
                    size=(m2, 1)
                )

            layer['v'] = 1 / np.sqrt(m2) * np.ones((m2, 1))

            layer['gamma'] = np.ones(shape=(m2, 1))
            layer['beta'] = np.zeros(shape=(m2, 1))
            
            self.layers.append(layer)
            
        del self.layers[-1]['gamma'], self.layers[-1]['beta']
     
    def initBatchNorm(
            self,
            X: np.array
        ) -> None:
        """
        Parameters
        ----------
        X : Nxd data matrix

        Returns
        -------
        None
        """
        _, _, _, _, muList, vList = self.evaluate(X, train=True)
        for mu, v, layer in zip(muList, vList, self.layers[:-1]):
            layer['mu'] = mu
            layer['v'] = v
    
    def updateBatchNorm(
            self,
            muList: list,
            vList: list
        ) -> None:
        """
        Parameters
        ----------
        muList : list w. computed batchNorm mean per layer
        vList : list w. computed batchNorm variance per layer
        Returns
        -------
        None
        """
        for mu, v, layer in zip(muList, vList, self.layers[:-1]):  
            layer['mu'] = self.alpha * layer['mu'] + (1 - self.alpha) * mu
            layer['v'] = self.alpha * layer['v'] + (1 - self.alpha) * v
     
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
        hList = [X.T.copy()]
        sList = []
        sListNorm = []
        muList = []
        vList = []
        
        for layer in self.layers[:-1]:
            s = layer['W'] @ hList[-1] + layer['b']  
            sList.append(s.copy())

            if train:
                mu = np.mean(s, axis=1)[..., np.newaxis]
                v = np.var(s, axis=1)[..., np.newaxis]
                
                muList.append(mu)
                vList.append(v)
            else:
                mu = layer['mu']
                v = layer['v']
            
            # s = np.power(v + 1e-12, -0.5) * np.eye(v.shape[0]) @ (s - mu)
            s = (s - mu) / np.sqrt(v + 1e-12) 
            sListNorm.append(s.copy())
            
            s = layer['gamma'] * s + layer['beta']
            hList.append(np.maximum(0, s))
        
        # update params for batchNorm
        self.updateBatchNorm(muList, vList)
        
        s = self.layers[-1]['W'] @ hList[-1] + self.layers[-1]['b']
        P = softMax(s)
        
        if not train:
            return P
        else:
            return P, hList, sList, sListNorm, muList, vList
    
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
        
        # evaluate loss term
        l = - D**-1 * np.sum(Y.T * np.log(P))
        
        # evaluate regularization term
        r = 0
        for layer in self.layers:
            r += lambd * (np.sum(np.square(layer['W'])) + np.sum(np.square(layer['b'])))
      
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
        preds = self.predict(X)
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
        P, hList, _, _, _, _ = self.evaluate(X, train=True)
        g = -(Y.T - P) # K x N

        # init grads list
        gradsList = []
        
        # iteratively compute grads per layer
        for layer, h in zip(self.layers[::-1], hList[::-1]):
            W_grads = D**-1 * g @ h.T + 2 * lambd * layer['W']
            b_grads = D**-1 * np.sum(g, axis=1)
            b_grads = np.expand_dims(b_grads, axis=1)
            
            # save grads
            gradsList.append({
                'W':W_grads,
                'b':b_grads
            })
            
            # propagate g
            h[h > 0] = 1
            g = layer['W'].T @ g * h
        
        return gradsList[::-1]
    
    def computeGradsBatchNorm(
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
        P, hList, sList, sListNorm, muList, vList = self.evaluate(X, train=True)
        g = -(Y.T - P) # K x N

        # init grads list
        gradsList = []
        
        # get h
        h = hList[-1].copy()
        
        # compute gradients for last layer
        W_grads = D**-1 * g @ h.T + 2 * lambd * self.layers[-1]['W']
        b_grads = D**-1 * np.sum(g, axis=1)
        
        # save grads
        gradsList.append({
            'W':W_grads,
            'b':b_grads[:, np.newaxis]
        })
        
        # propagate g to next layer
        h[h > 0] = 1                                
        g = self.layers[-1]['W'].T @ g * h
        
        # create zipped object for iterating over params
        iterObj = zip(
            self.layers[-2::-1],
            hList[-2::-1],
            sList[::-1],
            sListNorm[::-1],
            muList[::-1],
            vList[::-1]
        )
        
        # iteratively compute grads for remaining layers w. batchNorm
        for layer, h, s, sNorm, mu, v in iterObj:
            
            #get ones
            ones = np.ones((1, D))
            
            # compute grads and offset params
            gamma_grads = D**-1 * np.sum(g * sNorm, axis=1)
            beta_grads = D**-1 * np.sum(g, axis=1)
            
            # propagate grads through scaling            
            g = g * (layer['gamma'] @ ones)
            
            # propagate g through BatchNorm
            sigma1 = np.power(v + 1e-12, -0.5)
            sigma2 = np.power(v + 1e-12, -1.5)
            g1 = g * (sigma1 @ ones)
            g2 = g * (sigma2 @ ones)
            d = s - (mu @ ones)
            c = (g2 * d) @ ones.T 
            g = g1 - D**-1 * ((g1 @ ones.T) @ ones + d * (c @ ones))
            
            # compute weight grads
            W_grads = D**-1 * g @ h.T + 2 * lambd * layer['W']
            b_grads = D**-1 * np.sum(g, axis=1)
            
            # save grads
            gradsList.append({
                'W':W_grads,
                'b':b_grads[:, np.newaxis],
                'gamma':gamma_grads[:, np.newaxis],
                'beta':beta_grads[:, np.newaxis]
            })
            
            # propagate g to next layer
            h[h > 0] = 1
            g = layer['W'].T @ g * h
            
        return gradsList[::-1]
    
    def computeGradsNumerical(
            self, 
            X: np.array, 
            Y: np.array, 
            lambd: float,
            eps: float,
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
        gradsList = []
        
        for layerIdx, layer in enumerate(self.layers):
            layerDict = {}
            
            for name, weight in layer.items():
                shape = weight.shape
                w_perturb = np.zeros(shape)
                w_gradsNum = np.zeros(shape)
                w_0 = weight.copy()
                
                for i in range(self.K):
                    for j in range(min(shape[1], self.K)):
                # for i in range(shape[0]):
                #     for j in range(shape[1]):
                
                        # add perturbation
                        w_perturb[i, j] = eps
                        
                        # perturb weight vector negatively
                        # and compute cost
                        w_tmp = w_0 - w_perturb
                        self.layers[layerIdx][name] = w_tmp
                        _, cost1 = self.computeCost(X, Y, lambd)
                    
                        # perturb weight vector positively
                        # and compute cost
                        w_tmp = w_0 + w_perturb
                        self.layers[layerIdx][name] = w_tmp
                        _, cost2 = self.computeCost(X, Y, lambd)
                        lossDiff = (cost2 - cost1) / (2 * eps)
                        
                        # get numerical grad f. W[i, j]
                        w_gradsNum[i, j] = lossDiff
                        w_perturb[i, j] = 0
            
                # reset weigth vector
                self.layers[layerIdx][name] = w_0
                layerDict[name] = w_gradsNum
            gradsList.append(layerDict)
            
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
        grads = self.computeGradsBatchNorm(X, Y, lambd)
        
        for grad, layer in zip(grads[:-1], self.layers[:-1]):
            layer['W']      -= eta * grad['W']
            layer['b']      -= eta * grad['b']
            layer['gamma']  -= eta * grad['gamma']
            layer['beta']   -= eta * grad['beta']
            
        self.layers[-1]['W']      -= eta * grads[-1]['W']
        self.layers[-1]['b']      -= eta * grads[-1]['b']
        
        