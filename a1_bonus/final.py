
import os
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
    P : kxN probability matrix w. applied softmax activation
    """
    S = np.exp(S)
    return S / np.sum(S, axis=0)

def sigmoid(S: np.array) -> np.array:
    """
    Parameters
    ----------
    S : dxN score matrix

    Returns
    -------
    P : kxN probability matrix w. sigmoid activations
    """
    return 1 / (1 + np.exp(-S))

def multBCE(p: np.array, y: np.array, K: int) -> np.array:
    """
    Parameters
    ----------
    P : kx1 probability vector
    Y : 1xk one-hot encoded vectro
    
    Returns
    -------
    l : multiple binary cross-entropy loss f. (p(x), y)
    """
    ones = np.ones(K)
    l = - 1 / K * (np.dot(ones - y, np.log(ones - p)) + np.dot(y, np.log(p)))
    return l

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
 
def getWeightImg(
        W: np.array
    ) -> list:
    """
    Parameters
    ----------
    W: Kxd weight matrix
    
    Returns
    -------
    list w. "plottable" weights
    """
    wList = []
    for k in range(len(W)):
        
        img = W[k, :].reshape(3, 32, 32).transpose(1, 2, 0)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        wList.append(img)
        
    return wList
       
def imgFlip(X: np.array, prob: float) -> np.array:
    """
    Parameters
    ----------
    X : nxd flattened img. array
    angle : int

    Returns
    -------
    X : nxd shuffled, flattened img. array w. some flipped inputs
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
    
    # X_flipped = X_flipped.transpose(0, 2, 3, 1)
    # X_flipped = np.flip(X_flipped, axis=2)
    # X_flipped = X_flipped.transpose(0, 3, 1, 2).reshape((N, d))
    
    # concatenate back into one array
    X[idxs] = X_flipped
    
    return X

class linearClassifier:
    def __init__(
            self, 
            K: int, 
            d: int,
            activation: str,
            seed: int
        ):
        
        assert (activation in ['sigmoid', 'softmax']), \
            'activation has to be sigmoid or softmax'
        
        # init weight dims
        self.K = K
        self.d = d
        
        # init activation type
        self.activation = activation
        
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
        S = np.matmul(self.W, X.T) + self.b
        
        # return scores w. softmax activation
        if self.activation == 'softmax':
            return softMax(S)
        else:
            return sigmoid(S)
    
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
        J : cross-entropy loss w. L2-regularization
        """
        # get size of batch
        D = len(X)
        
        # get probabilities
        P = self.evaluate(X)
        
        # evaluate loss and regularization term
        if self.activation == 'softmax':
            l = 1 / D * sum([-np.dot(Y[i, :], np.log(P[:, i])) for i in range(D)])
            r = lambd * np.sum(np.square(self.W))
        else:
            l = 1 / D * sum([multBCE(P[:, i], Y[i, :], self.K) for i in range(D)]) 
            r = lambd * np.sum(np.square(self.W))
            
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
        
        # get weight gradients
        W_grads = 1 / D * np.matmul(g, X)
        b_grads = 1 / D * np.sum(g, axis=1)
        
        # # model dependent changes
        # if self.activation == 'softmax':
        #     W_grads += 2 * lambd * self.W
        # else:
        #     W_grads *= self.K**-1
        #     b_grads *= self.K**-1
    
        if self.activation == 'sigmoid':
            W_grads *= self.K**-1
            b_grads *= self.K**-1
    
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
        self.W = self.W - eta * W_grads
        self.b = self.b - eta * b_grads

def main():
    # define paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a1\\'
    plot_path = home_path + '\\a1_bonus\\plots\\'
    model_path = home_path + '\\a1_bonus\\models\\'
    
    ###1
    train_files = [
        'data_batch_1',
        'data_batch_2',
        'data_batch_3',
        'data_batch_4',
        'data_batch_5'
    ]
    
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
    X_train, X_val = X_train[:-1000], X_train[-1000:]
    k_train, k_val = k_train[:-1000], k_train[-1000:]
    Y_train, Y_val = Y_train[:-1000], Y_train[-1000:]
    
    ### 2
    # whiten w. training data
    mean_train = np.mean(X_train, axis=0)
    std_train  = np.std(X_train, axis=0)
    
    X_train = (X_train - mean_train) / std_train
    X_test  = (X_test - mean_train) / std_train
    X_val   = (X_val - mean_train) / std_train
    
    ### 3
    linearModel = linearClassifier(
        K=Y_train.shape[1], 
        d=X_train.shape[1],
        # activation='sigmoid',
        activation='sigmoid',
        seed=400
    )
    
    # ###
    # W_grads, b_grads = linearModel.computeGrads(
    #     X=X_train[:100], 
    #     Y=Y_train[:100], 
    #     lambd=0.1
    # )
    
    # ###
    # W_gradsNum, b_gradsNum = linearModel.computeGradsNumerical(
    #     X=X_train[:100], 
    #     Y=Y_train[:100], 
    #     lambd=0.1,
    #   eps = 1e-5
    # )
    
    # ### test gradient diff
    # W_gradDiffMax = np.max(np.abs(W_grads - W_gradsNum))
    # b_gradDiffMax = np.max(np.abs(b_grads - b_gradsNum))
    
    # ### print gradient diff
    # print(W_gradDiffMax, b_gradDiffMax)
    # # raise SystemExit(0)
    
    ### train model
    version     = 'v9.2'
    func        = 'sigmoid'
    lambd       = 0.1
    eta         = 0.01
    n_batch     = 100
    n_epochs    = 50
    n_decay     = 50
    flip_p      = 0.5
    
    # create lists for storing results
    trainLoss, valLoss, trainCost, valCost, testAcc = [], [], [], [], []
    
    # create list of idxs for shuffling
    idxs = list(range(len(X_train)))
    
    # print out parameter settings
    print('PARAMETERS: \n\t Version: {}\n\t Activation: {}\n\t Lambda: {:.2f}\n\t Eta: {:.2f}\n\t N_batch: {}\n\t N_epochs: {}\n\t N_decay: {}\n\t P_flip: {:.2f}\n\n'.format(
        version,
        func,
        lambd,
        eta,
        n_batch,
        n_epochs,
        n_decay,
        flip_p
    ))
    
    for epoch in range(n_epochs):
        # shuffle training examples
        np.random.shuffle(idxs)
        X_train, Y_train = X_train[idxs], Y_train[idxs]
        
        # # flip images
        if flip_p > 0:
            X_train = imgFlip(X_train, prob=flip_p)
        
        # decay learning rate
        if epoch % n_decay == 0:
            eta *= 0.5
        
        # iterate over batches
        for i in range(len(X_train) // n_batch):
            X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
            Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
            
            linearModel.train(
                X=X_trainBatch, 
                Y=Y_trainBatch, 
                lambd=lambd, 
                eta=eta
            )
            
        # compute cost and loss
        epochTrainLoss, epochTrainCost = linearModel.computeCost(
            X_train, 
            Y_train, 
            lambd=lambd
        )
        epochValLoss, epochValCost     = linearModel.computeCost(
            X_val, 
            Y_val, 
            lambd=lambd
        )
        
        # store vals
        trainLoss.append(epochTrainLoss)
        trainCost.append(epochTrainCost)
        valLoss.append(epochValLoss)
        valCost.append(epochValCost)
        
        # compute accuracy on test set
        testAcc.append(linearModel.computeAcc(X_test, k_test))
        
        print(
            'EPOCH {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.4f}'\
            .format(epoch, trainLoss[-1], valLoss[-1], testAcc[-1])
        )
            
    # plot COST function
    plt.plot(trainCost, 'b', linewidth=1.5, alpha=1.0, label='Training')
    plt.plot(valCost, 'r', linewidth=1.5, alpha=1.0, label='Validation')
    
    plt.xlim(0, n_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost', rotation=0, labelpad=20)
    plt.title('Cost per training epoch')
    plt.legend(loc='upper right')
    #plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot LOSS function
    plt.plot(trainLoss, 'b', linewidth=1.5, alpha=1.0, label='Training')
    plt.plot(valLoss, 'r', linewidth=1.5, alpha=1.0, label='Validation')
    
    plt.xlim(0, n_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss', rotation=0, labelpad=20)
    plt.title('Loss per training epoch')
    plt.legend(loc='upper right')
    #plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot ACCURACY
    plt.plot([acc * 100 for acc in testAcc], 'm', linewidth=2.5, alpha=1.0)
    plt.xlim(0, n_epochs)
    plt.xlabel('Epoch')
    plt.ylabel('%', rotation=0, labelpad=20)
    plt.title('Testing accuracy')
    #plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot WEIGHTS
    wImgs = getWeightImg(linearModel.W)
    _, axs = plt.subplots(1, 10)
    for img, ax in zip(wImgs, axs.flatten()):
        ax.imshow(img)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax.tick_params(left=False, bottom=False)
    
    #plt.savefig(plot_path + 'weights_{}.png'.format(version), bbox_inches='tight', dpi=500)
    plt.show()
    
    # save MODEL
    # saveAsMat(linearModel.W, model_path + 'model_{}_W'.format(version))
    # saveAsMat(linearModel.b, model_path + 'model_{}_b'.format(version))
    
if __name__ == '__main__':
    main()