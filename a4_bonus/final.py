
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    numCats = np.max(k)
    return np.array([[
        1 if idx == label else 0 for idx in range(numCats+1)]
         for label in k]
    )

class AdamOpt:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            weights: list,
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init moments
        self.m = {}
        self.v = {}
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
            self.v[name] = np.zeros(weight.shape)
            
    def calcMoment(self, beta, moment, grad):
        newMoment = beta * moment + (1 - beta) * grad
        return newMoment
    
    def step(self, weight, grad, t):     
        # update fist moment and correct bias
        self.m[weight] = self.calcMoment(
            self.beta1 ** t,
            self.m[weight], 
            grad
        )
        
        # update second moment and correct bias
        self.v[weight] = self.calcMoment(
            self.beta2 ** t,
            self.v[weight], 
            np.square(grad)
        )
        
        mCorrected = self.m[weight] / (1 - self.beta1 ** t + self.eps)
        vCorrected = self.v[weight] / (1 - self.beta2 ** t + self.eps)
        stepUpdate = mCorrected / (np.sqrt(vCorrected) + self.eps)
        
        return stepUpdate
    
class AdaGrad:
    def __init__(
            self,
            eps: float,
            weights: list
        ):
        # save init params
        self.eps = eps
        
        # init dicts for saving moments
        self.m = {}
        
        # init moments
        for name, weight in weights.items():
            self.m[name] = np.zeros(weight.shape)
    
    def step(self, weight, grad, t):
        
        self.m[weight] += np.square(grad)
        stepUpdate = grad / np.sqrt(self.m[weight] + self.eps)
        
        return stepUpdate

class RNN:
    def __init__(
            self, 
            K: int, 
            m: list,
            sigma: float,
            optimizer: str,
            initialization: str,
            seed: int
        ):
        
        # init seed
        np.random.seed(seed)
        
        # init weight dims
        self.K = K
        self.m = m
        
        # init weight dict
        self.weights = {}
        
        # init bias/shift weights
        self.weights['b'] = np.zeros(shape=(self.m, 1))
        self.weights['c'] = np.zeros(shape=(self.K, 1))
        
        # init weight matrices
        if initialization.lower() == 'he':
            self.weights['U'] = np.random.normal(loc=0, scale=2/np.sqrt(self.m), size=(self.m, self.K))
            self.weights['W'] = np.random.normal(loc=0, scale=2/np.sqrt(self.m), size=(self.m, self.m))
            self.weights['V'] = np.random.normal(loc=0, scale=2/np.sqrt(self.m), size=(self.K, self.m))
        else:
            self.weights['U'] = np.random.randn(self.m, self.K) * sigma
            self.weights['W'] = np.random.randn(self.m, self.m) * sigma
            self.weights['V'] = np.random.randn(self.K, self.m) * sigma
        
        
        # initialize hprev
        self.hprev = np.zeros(shape=(self.m, 1))

        # init optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = AdamOpt(
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weights=self.weights
            )
        else:
            self.optimizer = AdaGrad(
                eps=1e-8,
                weights=self.weights
            )

    def synthesizeText(
            self,
            x0: np.array,
            n: int,
            T: float,
            theta: float
        ) -> list:
        
        h = self.hprev
        xList = [x0]
        for _ in range(n):
            a = self.weights['W'] @ h + self.weights['U'] @ xList[-1].T + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o / T)
            
            # nucleus sampling
            if theta > 0:
                pt = list(enumerate(np.squeeze(p)))
                pt.sort(key=lambda pair: pair[-1], reverse=True)
                cutoff = np.argmax(np.cumsum([pair[-1] for pair in pt]) > theta)
                
                probVec = np.array([pair[-1] for pair in pt[:cutoff+1]])
                probVec /= np.sum(probVec) 
                
                idxNext = np.random.choice(
                    a=range(cutoff+1), 
                    p=probVec
                )
                
                idxNext = pt[idxNext][0]
            
            else:    
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
        for x in X:
            a = self.weights['W'] @ hList[-1] + self.weights['U'] @ x[:, np.newaxis] + self.weights['b']
            h = np.tanh(a)
            o = self.weights['V'] @ h + self.weights['c']
            p = softMax(o)
            
            # save vals
            aList.append(a)
            hList.append(h)
            oList.append(o)
            probSeq.append(p)
        
        P = np.hstack(probSeq)
        A = np.hstack(aList)
        H = np.hstack(hList)
        O = np.hstack(oList)
        
        # update hprev
        if train:
            self.hprev = H[:, -1][:, np.newaxis]
        
        if train:
            # return P, aList, hList, oList
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
        # get probs
        P = self.evaluate(X, train=False)
        
        # evaluate loss term
        l = - np.sum(Y.T * np.log(P))
        
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
            Y: np.array
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
        g = -(Y.T - P)
        
        # get V grad
        V_grads = g @ H.T[1:]
        c_grads = np.sum(g, axis=1)[:, np.newaxis]
        
        # compute grads for a and h
        h_grad = g.T[-1] @ self.weights['V']
        a_grad = h_grad * (1 - np.square(np.tanh(A.T[-1])))
        
        # init lists for grads, a and h
        h_grads = [h_grad]
        a_grads = [a_grad]
        
        for g_t, a_t in zip(g.T[-2::-1], A.T[-2::-1]):
            
            h_grad = g_t @ self.weights['V'] + a_grad @ self.weights['W']
            a_grad = h_grad * (1 - np.square(np.tanh(a_t)))
        
            h_grads.append(h_grad)
            a_grads.append(a_grad)
        
        h_grads = np.vstack(h_grads[::-1]).T
        a_grads = np.vstack(a_grads[::-1]).T
        
        # get W grads
        W_grads = a_grads @ H.T[:-1]
        U_grads = a_grads @ X
        b_grads = np.sum(a_grads, axis=1)[:, np.newaxis]
        
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
            t: int,
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
        grads = self.computeGrads(X, Y)
        for key, weight in self.weights.items():
            # clip gradient
            grads[key] = np.clip(grads[key], -5, 5)
            
            # get update
            stepUpdate = self.optimizer.step(key, grads[key], t)
            
            # update weight
            weight -= eta * stepUpdate

def main():

    # get paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a4\\'
    plot_path = home_path + '\\a4_bonus\\plots\\'
    models_path = home_path + '\\a4_bonus\\models\\'
    results_path = home_path + '\\a4_bonus\\results\\'
    
    # get text data
    fname = 'goblet_book.txt'
    fpath = data_path + fname
    
    # read text file
    with open(fpath, 'r') as fo:
        data = fo.readlines()
    
    # spec certain chars to remove
    removeList = ['0', '1', '2', '3', '4', '6', '7', '9', '}', '¢', '¼', 'Ã', 'â', '€']
    
    # split lines into words and words into chars
    data = [char 
                for line in data
                    for word in list(line)
                        for char in list(word)
                            if char not in removeList
    ]
    
    # create word-key-word mapping
    keyToChar = dict(enumerate(np.unique(data)))
    charToKey = dict([(val, key) for key, val in keyToChar.items()])
    
    # define data specs
    seq_length = 50
    
    # define X, w. one-hot encoded representations
    data = oneHotEncode(np.array([charToKey[char] for char in data]))
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].astype('int8'))
    
    # partition X into blocks, get validation block
    n_blocks = 100
    block_size = len(X) // n_blocks
    X_blocks = [X[block_size*i:block_size*(i+1)] for i in range(n_blocks-1)]
    
    # # choose random blocks for validation
    # idxs = np.random.choice(range(n_blocks), size=10)
    # val_block = []
    # for count, idx in enumerate(idxs):
    #     val_block.append(X_blocks.pop(idx-count))
    X_blocks, val_block = X_blocks[:-5], X_blocks[-5:]
    val_block = np.vstack(val_block)
    
    # init networks
    version = 'v7'
    K  = len(keyToChar)
    m = 200
    sigma = 0.01
    initialization = 'He'
    optimizer = 'adam'
    eta = 0.001
    diverse = True
    
    recurrentNet = RNN(
        K=K,
        m=m,
        sigma=sigma,
        initialization=initialization,
        optimizer=optimizer,
        seed=2
    )
    
    # save best weights
    weights_best = recurrentNet.weights.copy()
    
    # init loss metrics
    trainLossHist = []
    valLossHist = []
    loss_smooth = seq_length * 4
    loss_best = loss_smooth
    
    valLoss_smooth = seq_length * 4
    
    # init global counters
    t = 0
    block_n = 0
    epoch_n = 0
    
    print ('\n------EPOCH {}--------\n'.format(epoch_n))
    while epoch_n < 10:
        if diverse:
            np.random.shuffle(X_blocks)
        
        for block in X_blocks:
            # reset hPrev
            if diverse:
                recurrentNet.hprev = np.zeros(shape=(m, 1))
            
            # reset counter
            e = 0
            while e < (block_size - seq_length):
                # train on sequence
                recurrentNet.train(block[e], block[e+1], block_n, eta=eta)
                loss, _ = recurrentNet.computeCost(block[e], block[e+1], lambd=0)
                loss_smooth = 0.999 * loss_smooth + 0.001 * loss
                
                # get random validation example
                valIdx = np.random.randint(len(val_block))-1
                valLoss, _ = recurrentNet.computeCost(val_block[valIdx], val_block[valIdx+1], lambd=0)
                valLoss_smooth = 0.999 * valLoss_smooth + 0.001 * valLoss
                
                # save loss metrics
                trainLossHist.append(loss_smooth)
                valLossHist.append(valLoss_smooth)
                
                # save weights if best loss
                if loss_smooth < loss_best:
                    weights_best = recurrentNet.weights.copy()
                    loss_best = loss_smooth
                    
                # print generated sequence
                if t % 10000 == 0:
                    sequence = recurrentNet.synthesizeText(
                        x0=block[e+1][:1], 
                        n=250,
                        T=1.0,
                        theta=0.6
                    )
                    
                    # convert to chars and print sequence
                    sequence = ''.join([keyToChar[key] for key in sequence])
                    print('\nGenerated sequence \n\n {}\n'.format(sequence))
                
                # update e, t
                t += 1
                e += seq_length
        
            
            # compute validation loss
            # reset hPrev
            if diverse:
                recurrentNet.hprev = np.zeros(shape=(m, 1))
        
            # print metrics
            print('Step {}, block: {}, train LOSS: {:.2f}, val LOSS: {:.2f}'.format(
                t, 
                block_n,
                loss_smooth,
                valLoss_smooth
            ))
            
            block_n += 1
            
        # update EPOCH
        epoch_n += 1
        print('\n------EPOCH {}--------\n'.format(epoch_n))
            
        # reset hPrev
        recurrentNet.hprev = np.zeros(shape=(m, 1))
              
    # plot results
    steps = [step * block_size for step in range(len(trainLossHist))]
    plt.plot(steps, trainLossHist, 'r', linewidth=1.5, alpha=1.0, label='Training')
    plt.plot(steps, valLossHist, 'g', linewidth=1.5, alpha=1.0, label='Validation')
    plt.xlim(0, steps[-1])
    plt.ylim(1.5,4.0)
    plt.xlabel('Steps')
    plt.ylabel('', rotation=0, labelpad=20)
    plt.title('Smooth loss for $10$ epochs')
    plt.legend(loc='upper right')
    # plt.savefig(plot_path + 'rnn_loss_{}.png'.format(version), dpi=200)
    plt.show()
    
    # # save model
    # with open(models_path + 'model_{}'.format(version), 'wb') as fo:
    #     pickle.dump(recurrentNet.weights, fo)
    
    # # save results
    # with open(results_path + 'loss_{}'.format(version), 'wb') as fo:
    #     pickle.dump({
    #         'train':trainLossHist,
    #         'val':valLossHist
    #         },
    #         fo
    #     )
    
    # recurrentNet.weights = weights_best
    sequence = recurrentNet.synthesizeText(
        x0=X_blocks[0][0][:1], 
        n=300,
        T=1.0,
        theta=1.0
    )
    
    # convert to chars and print sequence
    sequence = ''.join([keyToChar[key] for key in sequence])
    print('\nGenerated sequence \n\n {}\n'.format(sequence))
    
if __name__ == '__main__':
    main()