
import os
import json
import pickle
import scipy.io as sio
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

class AdamOpt:
    def __init__(
            self,
            beta1: float,
            beta2: float,
            eps: float,
            weights: list
        ):
        # save init params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # init dicts for saving moments
        self.m, self.v = {}, {}
        
        # init moments
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
        
        mCorrected = self.m[weight] / (1 - self.beta1 ** t)
        vCorrected = self.v[weight] / (1 - self.beta2 ** t)
        stepUpdate = mCorrected / (np.sqrt(vCorrected) + self.eps)
        return stepUpdate

class neuralNetwork:
    def __init__(
            self, 
            K: int, 
            d: int,
            m: int,
            p_dropout: float,
            optimizer: str,
            seed: int
        ):
        
        # init weight dims
        self.K = K
        self.d = d
        self.m = m
        
        # init dropout 
        self.p_dropout = p_dropout
        
        # init optimizer
        self.optimizer = optimizer
        
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
        
        # init optimizer
        if self.optimizer == 'Adam':
            self.opt = AdamOpt(
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weights=self.weights
            )
        
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
        # get binary mask for dropout
        if train and (self.p_dropout > 0):
            mask = np.random.binomial(
                n=1,
                p=1-self.p_dropout,
                size=(1,self.m)
            )
        else:
            mask = np.ones(shape=(1,self.m))
            
        # calculate forward pass
        s1 = self.weights['W1'] @ X.T + self.weights['b1'] # m x N
        h = np.maximum(0, s1)  # m x N
        h = (1 - train * self.p_dropout)**-1 * mask.T * h
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
            eta: float, 
            t: int
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
        for grad, weight in zip(grads, self.weights):
            
            if self.optimizer == 'Adam':
                stepUpdate = self.opt.step(weight, grad, t)
                self.weights[weight] -= eta * stepUpdate
            else:
                self.weights[weight] -= eta * grad

def trainNetwork(
        n_epochs: int, 
        n_batch: int, 
        eta_min: float, 
        eta_max: float, 
        ns: int, 
        n_cycles: int, 
        lambd: float,
        m: int,
        p_dropout: float,
        p_flip: float,
        p_transl: float,
        optimizer: str,
        plot: bool,
        version: str
    ):

    # define paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a1\\'
    plot_path = home_path + '\\a2_bonus\\plots\\'
    
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

    # INIT model
    neuralNet = neuralNetwork(
        K = 10,
        d = 3072,
        m = m,
        p_dropout = p_dropout,
        optimizer = optimizer,
        seed=1
    )
    
    # init time step
    t = 0
    
    # init lists/dicts
    etaHist, accHist = [], []
    lossHist, costHist = {'train':[], 'val':[]}, {'train':[], 'val':[]}
    
    # create list of idxs for shuffling
    idxs = list(range(len(X_train)))
    
    for epoch in range(1, n_epochs+1):
        # shuffle training examples
        np.random.shuffle(idxs)
        X_train, Y_train, k_train = X_train[idxs], Y_train[idxs], k_train[idxs]
        
        # flip images
        if (p_flip > 0) or (p_transl > 0):
            X_augm = X_train.copy()
            X_augm = imgFlip(X_augm, prob=p_flip)
            X_augm = imgTransl(X_augm, prob=p_transl)
        
        # iterate over batches
        for i in range(len(X_train) // n_batch):
            X_trainBatch = X_augm[i*n_batch:(i+1)*n_batch]
            Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
            
            # update eta
            if optimizer == 'Adam':
                eta = 0.001
            else:
                eta = cyclicLearningRate(
                    etaMin=eta_min, 
                    etaMax=eta_max, 
                    stepSize=ns, 
                    timeStep=t
                )
            
            # run training, GD update
            neuralNet.train(
                X=X_trainBatch, 
                Y=Y_trainBatch, 
                lambd=lambd, 
                eta=eta,
                t=epoch
            )
            
            # append to list of eta and update time step
            etaHist.append(eta)
            t += 1
            
            # if some number of cycles, break
            if t >= n_cycles * 2 * ns:
                break
        
            # add loss, cost info, 4 times per cycle
            if t % (ns / 10) == 0:
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
                
                # get acc
                acc = neuralNet.computeAcc(X_test, k_test)
                
                # save info
                lossHist['train'].append(trainLoss)
                lossHist['val'].append(valLoss)
                costHist['train'].append(trainCost)
                costHist['val'].append(valCost)
                accHist.append(acc)
            
                # print info
                print(
                    '\t STEP {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.4f}'\
                    .format(t, lossHist['train'][-1], costHist['train'][-1], accHist[-1])
                )
    
    if not plot:
        return lossHist, costHist, accHist
    else:
        # define steps for plot
        steps = [step * (ns / 10) for step in range(len(costHist['train']))]
        
        # plot COST function
        plt.plot(steps, costHist['train'], 'g', linewidth=1.5, alpha=1.0, label='Training')
        plt.plot(steps, costHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
        
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('Cost', rotation=0, labelpad=20)
        plt.title('Cost')
        plt.legend(loc='upper right')
        #plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
        plt.show()
        
        # plot LOSS function
        plt.plot(steps, lossHist['train'], 'g', linewidth=1.5, alpha=1.0, label='Training')
        plt.plot(steps, lossHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
        
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('Loss', rotation=0, labelpad=20)
        plt.title('Loss')
        plt.legend(loc='upper right')
        #plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
        plt.show()
        
        # plot ACCURACY
        plt.plot(steps, [acc * 100 for acc in accHist], 'b', linewidth=2.5, alpha=1.0)
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('%', rotation=0, labelpad=20)
        plt.title('Testing accuracy')
        #plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
        plt.show()
        
        return lossHist, costHist, accHist

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a2_bonus\\results\\'

# set filename
fname = 'training_v3'
fpath = results_path + fname

# set params
n_epochs = 50
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
ns      = 2 * 45000 // n_batch
n_cycles = 2
p_flip = 0.4
p_transl = 0.05

# specify network
optimizer = 'Adam'
# optimizer = ''

# set lambda values
lambdas = [
    0.00005,
    0.0001,
    0.001
]

mList = [
    50,
    75,
    100,
    125,
    150
]

pDropList = [
    0.0,
    0.2,
    0.3,
    0.4,
    0.5
]

# init dictionary for saving
saveDict = {
    'params':{
        'n_epochs':n_epochs,
        'n_batch':n_batch,
        'eta_min':eta_min,
        'eta_max':eta_max,
        'ns':ns,
        'n_cycles':n_cycles,
        #'m':m,
        #'p_dropout':p_dropout,
        'p_flip':p_flip,
        'p_transl':p_transl,
        'optimizer':optimizer
}}

# iterate over possible lambda values
for v1, lambd in enumerate(lambdas):
    for v2, m in enumerate(mList):
        for v3, pDrop in enumerate(pDropList):
            # get version
            version = 'v' + str(v1) + str(v2) + str(v3)
            
            print('\n TRAIN NETWORK ({}): cycles: {}, lambda: {:.5f}, m: {}, pDrop: {:.2f}\n'.format(
                version,
                n_cycles,
                lambd,
                m,
                pDrop
            ))
            
            # get training results
            lossHist, costHist, accHist = trainNetwork(
                n_epochs=n_epochs,
                n_batch=n_batch,
                eta_min=eta_min,
                eta_max=eta_max,
                ns=ns,
                n_cycles=n_cycles,
                m=m,
                p_dropout=pDrop,
                p_flip=p_flip,
                p_transl=p_transl,
                optimizer=optimizer,
                lambd=lambd,
                version=version,
                plot=False
            )
            
           # save version-specific results in dictionary
            saveDict[version] = {
                'lambda':lambd,
                'm':m,
                'pDrop':pDrop,
                'lossHist':lossHist,
                'costHist':costHist,
                'accHist':accHist
            }
            
            # dump results to JSON
            with open(fpath, 'w') as fp:
                json.dump(saveDict, fp)