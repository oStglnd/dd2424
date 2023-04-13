
import os
import numpy as np
import matplotlib.pyplot as plt

# get files
from model import neuralNetwork
from misc import getCifar, cyclicLearningRate

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
        plot: bool,
        version: str
    ):

    # define paths
    home_path = os.path.dirname(os.getcwd())
    data_path = home_path + '\\data\\a1\\'
    plot_path = home_path + '\\a2\\plots\\'
    
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
        
        # iterate over batches
        for i in range(len(X_train) // n_batch):
            X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
            Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
            
            # update eta
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
                eta=eta
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
        plt.plot(steps, costHist['train'], 'b', linewidth=1.5, alpha=1.0, label='Training')
        plt.plot(steps, costHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
        
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('Cost', rotation=0, labelpad=20)
        plt.title('Cost')
        plt.legend(loc='upper right')
        plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
        plt.show()
        
        # plot LOSS function
        plt.plot(steps, lossHist['train'], 'b', linewidth=1.5, alpha=1.0, label='Training')
        plt.plot(steps, lossHist['val'], 'r', linewidth=1.5, alpha=1.0, label='Validation')
        
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('Loss', rotation=0, labelpad=20)
        plt.title('Loss')
        plt.legend(loc='upper right')
        plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
        plt.show()
        
        # plot ACCURACY
        plt.plot(steps, [acc * 100 for acc in accHist], 'm', linewidth=2.5, alpha=1.0)
        plt.xlim(0, steps[-1])
        plt.xlabel('Step')
        plt.ylabel('%', rotation=0, labelpad=20)
        plt.title('Testing accuracy')
        plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
        plt.show()
        
        return lossHist, costHist, accHist