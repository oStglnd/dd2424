
import numpy as np
from misc import softMax

class batchRNN:
    def __init__(
            self, 
            K: int, 
            m: list,
            batch_size: int,
            sigma: float,
            seed: int
        ):
        
        # init seed
        np.random.seed(seed)
        
        # init weight dims
        self.K = K
        self.m = m
        self.batch_size = batch_size
        
        # init weight dict
        self.weights = {}
        self.momentum = {}
        
        # init bias/shift weights
        self.weights['b'] = np.zeros(shape=(self.m, 1))
        self.weights['c'] = np.zeros(shape=(self.K, 1))
        
        # init weight matrices
        self.weights['U'] = np.random.randn(self.m, self.K) * sigma
        self.weights['W'] = np.random.randn(self.m, self.m) * sigma
        self.weights['V'] = np.random.randn(self.K, self.m) * sigma
        
        for key, weight in self.weights.items():
            self.momentum[key] = np.zeros(shape=weight.shape)
        
        # initialize hprev
        self.hprev = np.zeros(shape=(self.m, self.batch_size))

    def synthesizeText(
            self,
            x0: np.array,
            n: int
        ) -> list:
        
        h = np.zeros(shape=(self.m, 1))
        xList = [x0]
        for _ in range(n):
            a = self.weights['W'] @ h + self.weights['U'] @ xList[-1].T + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o)
            
            idxNext = np.random.choice(
                a=range(self.K), 
                p=np.squeeze(p)
            )
            
            x = np.zeros(shape=(1, self.K))
            x[0, idxNext] = 1
            xList.append(x)
        
        xList = [np.argmax(x) for x in xList]
        return xList
        

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
        probSeq = []        
        hList = [self.hprev.copy()]
        aList = []
        oList = []
        
        # iterate through recurrent block
        for x in X.T:
            a = self.weights['W'] @ hList[-1] + self.weights['U'] @ x + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o)
            
            # save vals
            aList.append(a)
            hList.append(h)
            oList.append(o)
            probSeq.append(p)
        
        P = np.stack(probSeq)
        A = np.stack(aList)
        H = np.stack(hList)
        O = np.stack(oList)
        
        if train:
            # return P, aList, hList, oList
            # self.hprev = H[-1]
            return P, A, H, O
        else:
            return P
    
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
            axis=1
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
        # get probs
        P = self.evaluate(X, train=False)
        
        # evaluate loss term
        l = - self.batch_size**-1 * np.sum(Y.T * np.log(P))
        
        # get regularization term
        r = 0
        
        return l, l + r
    
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
        return 0
    
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
        P, A, H, O = self.evaluate(X=X, train=True)
        G = -(Y.T - P)
        
        batchObj = list(zip(
            P.T,
            A.T,
            H.T,
            O.T,
            G.T,
            X
        ))
        
        V_grads, W_grads, U_grads, c_grads, b_grads = 0, 0, 0, 0, 0
        
        for p, a, h, o, g, x in batchObj:
        
            # get V grad
            V_grads += g @ h.T[1:]
            c_grads += np.sum(g, axis=1)[:, np.newaxis]
            
            # compute grads for a and h
            h_grad = g.T[-1] @ self.weights['V']
            a_grad = h_grad * (1 - np.square(np.tanh(a.T[-1])))
            
            # init lists for grads, a and h
            h_grads = [h_grad]
            a_grads = [a_grad]
            
            for g_t, a_t in zip(g.T[-2::-1], a.T[-2::-1]):
                
                h_grad = g_t @ self.weights['V'] + a_grad @ self.weights['W']
                a_grad = h_grad * (1 - np.square(np.tanh(a_t)))
            
                h_grads.append(h_grad)
                a_grads.append(a_grad)
            
            h_grads = np.vstack(h_grads[::-1]).T
            a_grads = np.vstack(a_grads[::-1]).T
            
            # get W grads
            W_grads += a_grads @ h.T[:-1]
            U_grads += a_grads @ x.T
            b_grads += np.sum(a_grads, axis=1)[:, np.newaxis]
        
        # save grads
        grads = {
            'V':V_grads,
            'W':W_grads,
            'U':U_grads,
            'b':b_grads,
            'c':c_grads
        }
        
        return grads
    
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
        gradsDict = {}

        for name, weight in self.weights.items():
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
                    self.weights[name] = w_tmp
                    _, cost1 = self.computeCost(X, Y, lambd)
                
                    # perturb weight vector positively
                    # and compute cost
                    w_tmp = w_0 + w_perturb
                    self.weights[name] = w_tmp
                    _, cost2 = self.computeCost(X, Y, lambd)
                    lossDiff = (cost2 - cost1) / (2 * eps)
                    
                    # get numerical grad f. W[i, j]
                    w_gradsNum[i, j] = lossDiff
                    w_perturb[i, j] = 0
        
            # save grads
            gradsDict[name] = w_gradsNum
            
            # reset weigth vector
            self.weights[name] = w_0
            
        return gradsDict
    
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
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -5, 5)
            
            # calculate momentum
            self.momentum[key] += np.square(grads[key])
            
            # update weight
            weight -= eta * self.batch_size**-1 * grads[key] / np.sqrt(self.momentum[key] + 1e-12)