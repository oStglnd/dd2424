
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
    
def cyclicLearningRate(
        etaMin: float,
        etaMax: float,
        stepSize: float,
        timeStep: int
    ) -> float:
    """
    Parameters
    ----------
    etaMin : minumum learning rate
    etaMax : maximum learning rate
    stepSize : step size
    timeStep : current step
    
    Returns
    -------
    eta : current learning rate
    """
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


def main():
    
    # set version
    version = 'v1'
    
    # set model param
    seed = 200                     
    K = 10                           
    d = 3072                               
    m = [200, 100]             
    optimizer = 'adam'
    batchNorm = True
    alpha = 0.8
    precise = False
    p_dropout = 0.0
    initialization = 'He'
    lambd = 0.01
    sigma = 0.01
    
    # set training params
    eta = 0.001
    n_epochs = 30
    n_batch = 100
    adaptive = True
    cyclical = False
    decay = False
    decay_rate = 1e-4
    eta_min = 1e-5
    eta_max = 1e-2
    ns      = 5 * 45000 // n_batch
    n_cycles = 3
    
    # define paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a1\\'
    plot_path = home_path + '\\a3_bonus\\plots\\'
    results_path = home_path + '\\a3_bonus\\results\\'
    models_path = home_path + '\\a3_bonus\\models\\'
    
    # define fnames
    train_files = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5'
    ]
    
    # get data
    X_train, k_train, Y_train = getCifar(data_path, train_files[0])
    for file in train_files[1:]:
        X_trainAdd, k_trainAdd, Y_trainAdd = getCifar(data_path, file)
        
        X_train = np.concatenate((X_train, X_trainAdd), axis=0)
        k_train = np.concatenate((k_train, k_trainAdd), axis=0)
        Y_train = np.concatenate((Y_train, Y_trainAdd), axis=0)
    
    # delete placeholders
    del X_trainAdd, k_trainAdd, Y_trainAdd
    
    # get test data
    X_test, k_test, Y_test = getCifar(data_path, 'test_batch')
    
    # get validation data
    X_train, X_val = X_train[:-5000], X_train[-5000:]
    k_train, k_val = k_train[:-5000], k_train[-5000:]
    Y_train, Y_val = Y_train[:-5000], Y_train[-5000:]
    
    ### 2
    # whiten w. training data
    mean_train = np.mean(X_train, axis=0)
    std_train  = np.std(X_train, axis=0)
    
    X_train = (X_train - mean_train) / std_train
    X_test  = (X_test - mean_train) / std_train
    X_val   = (X_val - mean_train) / std_train
    
    # generate model
    neuralNet = neuralNetwork(
        K = K,
        d = d,
        m = m,
        batchNorm = batchNorm,
        alpha = alpha,
        precise = precise,
        p_dropout = p_dropout,
        initialization = initialization,
        optimizer = optimizer,
        sigma=sigma,
        seed=seed
    )
    
    # init lists/dicts
    etaHist, accHist = [], []
    lossHist, costHist = {'train':[], 'val':[]}, {'train':[], 'val':[]}
    
    # create list of idxs for shuffling
    idxs = list(range(len(X_train)))
    
    # create timestep
    t = 0
    
    # init batchNorm params
    if not precise:
        neuralNet.initBatchNorm(X_train[:n_batch])
    else:
        neuralNet.initBatchNorm(X_train)
    
    for epoch in range(1, n_epochs+1):
        # shuffle training examples
        np.random.shuffle(idxs)
        X_train, Y_train, k_train = X_train[idxs], Y_train[idxs], k_train[idxs]
        
        # iterate over batches
        for i in range(len(X_train) // n_batch):
            X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
            Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
            
            # update eta
            if cyclical:
                eta = cyclicLearningRate(
                    etaMin=eta_min, 
                    etaMax=eta_max, 
                    stepSize=ns, 
                    timeStep=t
                )
            elif decay:
                eta = max(eta_max * np.exp(-decay_rate * t), eta_min) 
            else:
                eta = eta
            
            # run training, GD update
            neuralNet.train(
                X=X_trainBatch, 
                Y=Y_trainBatch, 
                lambd=lambd, 
                step=epoch,
                eta=eta
            )
            
            # append to list of eta and update time step
            t += 1
            
            # update batchNorm params
            if precise:
                if (i % 20 == 0) and (i > 0):
                    neuralNet.initBatchNorm(X_train[(i-20)*n_batch:i*n_batch])
        
        # get loss, cost after each epoch
        trainLoss, trainCost = neuralNet.computeCost(
            X_train, 
            Y_train, 
            lambd=lambd
        )
        
        valLoss, valCost = neuralNet.computeCost(
            X_val, 
            Y_val, 
            lambd=lambd
        )
        
        # get acc for test data
        if adaptive:
            batchNorm_params = [
                {'mu':layer['mu'], 'v':layer['v']} for layer in neuralNet.layers[1:]
            ]
            
            # # augment test data w. image flipping
            # X_test = imgFlip(X_test, prob=0.0)
            
            # RE-init on X_test
            neuralNet.initBatchNorm(imgFlip(X_test, prob=0.1))
            acc = neuralNet.computeAcc(X_test, k_test)
            
            # reset params
            for idx, params in enumerate(batchNorm_params):
                neuralNet.layers[idx+1]['mu'] = params['mu']
                neuralNet.layers[idx+1]['v'] = params['v']
        else:
            if precise:
                neuralNet.initBatchNorm(X_train)
            acc = neuralNet.computeAcc(X_test, k_test)
        
        # save info
        etaHist.append(eta)
        lossHist['train'].append(trainLoss)
        lossHist['val'].append(valLoss)
        costHist['train'].append(trainCost)
        costHist['val'].append(valCost)
        accHist.append(acc)
            
        # print info
        print(
            '\t EPOCH {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.4f}'\
            .format(epoch, lossHist['train'][-1], costHist['train'][-1], accHist[-1])
        )
    
    # # save model
    # with open(models_path + 'model_{}'.format(version), 'wb') as fo:
    #     pickle.dump(neuralNet.layers, fo)
    
    # # save results
    # with open(results_path + 'loss_{}'.format(version), 'wb') as fo:
    #     pickle.dump({
    #         'loss':lossHist,
    #         'cost':costHist,
    #         'acc':accHist,
    #         'eta':etaHist
    #         },
    #         fo
    #     )
                   
    # define steps for plot               
    # steps = [step * (ns / 10) for step in range(len(costHist['train']))]
    steps = list(range(n_epochs))
                 
    # plot COST function
    plt.plot(steps, costHist['train'], 'g', linewidth=1.5, alpha=1.0, label='Training')
    plt.plot(steps, costHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
    
    plt.xlim(0, steps[-1])
    # plt.ylim(0, max(costHist['train']) * 1.5)
    plt.ylim(0.5, 3.0)
    plt.xlabel('Epoch')
    plt.ylabel('Cost', rotation=0, labelpad=20)
    plt.title('Cost')
    plt.legend(loc='upper right')
    # plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot LOSS function
    plt.plot(steps, lossHist['train'], 'g', linewidth=1.5, alpha=1.0, label='Training')
    plt.plot(steps, lossHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
    
    plt.xlim(0, steps[-1])
    # plt.ylim(0, max(lossHist['train']) * 1.5)
    plt.ylim(0.5, 3.0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title('Loss')
    plt.legend(loc='upper right')
    # plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot ACCURACY
    plt.plot(steps, [acc * 100 for acc in accHist], 'b', linewidth=2.5, alpha=1.0)
    plt.ylim(20,60)
    plt.xlim(0, steps[-1])
    plt.xlabel('Epoch')
    plt.ylabel('%', rotation=0, labelpad=20)
    plt.title('Testing accuracy')
    # plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
    plt.show()
    
if __name__ == '__main__':
    main()