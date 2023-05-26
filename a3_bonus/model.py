

import numpy as np
from misc import softMax

class AdamOpt:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            layers: list,
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init dicts for saving moments
        self.m, self.v = [], []
        
        # init moments
        for idx, layer in enumerate(layers):
            self.m.append({})
            self.v.append({})
            for name, weight in layer.items():
                self.m[idx][name] = np.zeros(weight.shape)
                self.v[idx][name] = np.zeros(weight.shape)
            
    def calcMoment(self, beta, moment, grad):
        newMoment = beta * moment + (1 - beta) * grad
        return newMoment
    
    def step(self, layerIdx, weight, grad, t):     
        # update fist moment and correct bias
        self.m[layerIdx][weight] = self.calcMoment(
            self.beta1,
            self.m[layerIdx][weight], 
            grad
        )
        
        # update second moment and correct bias
        self.v[layerIdx][weight] = self.calcMoment(
            self.beta2,
            self.v[layerIdx][weight], 
            np.square(grad)
        )
        
        mCorrected = self.m[layerIdx][weight] / (1 - self.beta1 ** t + self.eps)
        vCorrected = self.v[layerIdx][weight] / (1 - self.beta2 ** t + self.eps)
        stepUpdate = mCorrected / (np.sqrt(vCorrected) + self.eps)
        return stepUpdate
    
class AdaGrad:
    def __init__(
            self,
            eps: float,
            layers: list
        ):
        # save init params
        self.eps = eps
        
        # init dicts for saving moments
        self.m = []
        
        # init moments
        for idx, layer in enumerate(layers):
            self.m.append({})
            for name, weight in layer.items():
                self.m[idx][name] = np.zeros(weight.shape)
    
    def step(self, layerIdx, weight, grad, t):
        
        self.m[layerIdx][weight] += np.square(grad)
        stepUpdate = grad / (np.sqrt(self.m[layerIdx][weight]) + self.eps)
        
        return stepUpdate

class neuralNetwork:
    def __init__(
            self, 
            K: int, 
            d: int,
            m: list,
            batchNorm: bool,
            alpha: float,
            precise: bool,
            p_dropout: float,
            initialization: str,
            optimizer: str,
            sigma: float,
            seed: int
        ):
        
        # init weight dims
        self.K = K
        self.d = d
        
        # init weight dims list
        weightList = [d] + m + [K]
        self.layers = []    
        
        # init batchNorm param
        self.batchNorm = batchNorm
        self.alpha = alpha
        self.precise = precise
        
        # init dropout 
        self.p_dropout = p_dropout
        
        # iterate over weight dims
        np.random.seed(seed)
        for m1, m2 in zip(weightList[:-1], weightList[1:]):
            layer = {}
            
            if initialization.lower() == 'he':
                scale = 2/np.sqrt(m1)
            else:
                scale = sigma
                
            layer['W'] = np.random.normal(
                    loc=0, 
                    scale=scale, 
                    size=(m2, m1)
            )
            
            layer['b'] = np.zeros(shape=(m2, 1))
            
            if self.batchNorm:
                for param in ['mu', 'beta']:
                    layer[param] = np.random.normal(
                        loc=0, 
                        scale=0.0, 
                        size=(m2, 1)
                    )
    
                for param in ['v', 'gamma']:
                    layer[param] = np.random.normal(
                        loc=1, 
                        scale=1/np.sqrt(m2), 
                        size=(m2, 1)
                    )
            
            self.layers.append(layer)
        
        if self.batchNorm:
            del self.layers[-1]['gamma'], self.layers[-1]['beta']
     
        
        # init optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = AdamOpt(
                beta1=0.9,
                beta2=0.999,
                eps=1e-12,
                layers=self.layers
            )
        elif optimizer.lower() == 'adagrad':
            self.optimizer = AdaGrad(
                eps=1e-12,
                layers=self.layers
            )
        else:
            self.optimizer = None
        
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
            # # get binary mask for dropout
            # if train and (self.p_dropout > 0):
            #     mask = np.random.binomial(
            #         n=1,
            #         p=1-self.p_dropout,
            #         size=layer['b'].shape
            #     )
            # else:
            #     mask = np.ones(shape=layer['b'].shape)
            
            s = layer['W'] @ hList[-1] + layer['b']  
            sList.append(s.copy())
            
            if self.batchNorm:
                if train:
                    mu = np.mean(s, axis=1, keepdims=True)#[..., np.newaxis]
                    v = np.var(s, axis=1, keepdims=True)#[..., np.newaxis]
                    
                    muList.append(mu)
                    vList.append(v)
                else:
                    mu = layer['mu']
                    v = layer['v']
                
                s = (s - mu) / np.sqrt(v + 1e-12) 
                sListNorm.append(s.copy())
                s = layer['gamma'] * s + layer['beta']
                
                # update params for batchNorm if NOT precise
                if not self.precise:
                    self.updateBatchNorm(muList, vList)
            
            h = np.maximum(0, s)
            # h = (1 - train * self.p_dropout)**-1 * mask * h
            hList.append(h)
        
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
            g = g1 - D**-1 * (g1 @ ones.T) @ ones - D**-1 * d * (c @ ones)
            
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
            step: int,
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
        if self.batchNorm:
            grads = self.computeGradsBatchNorm(X, Y, lambd)
        else:
            grads = self.computeGrads(X, Y, lambd)
        
        for layerIdx, (grad, layer) in enumerate(zip(grads, self.layers)):
            for weightKey, weightVals in layer.items():
                if weightKey not in ['mu', 'v']:
                    
                    if self.optimizer:
                        stepUpdate = self.optimizer.step(
                            layerIdx,
                            weightKey, 
                            grad[weightKey], 
                            step
                        )
                        weightVals -= eta * stepUpdate
                    else:
                        weightVals -= eta * grad[weightKey]