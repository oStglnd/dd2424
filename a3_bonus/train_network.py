
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from misc import getCifar, cyclicLearningRate, imgFlip
from model import neuralNetwork


def trainNetwork(
        version: str,
        m: list,
        optimizer: str,
        cyclical: bool,
        decay: bool):
    

    # set model param
    seed = 200                             # init seed
    K = 10                                 # init n. of classes
    d = 3072                               # init dimensions
    batchNorm = True
    alpha = 0.8
    precise = False
    p_dropout = 0.0
    initialization = 'He'
    lambd = 0.01
    sigma = 0.01
    
    # set training params
    eta = 0.001
    n_epochs = 50
    n_batch = 100
    adaptive = True
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
    
    # save model
    with open(models_path + 'model_{}'.format(version), 'wb') as fo:
        pickle.dump(neuralNet.layers, fo)
    
    # save results
    with open(results_path + 'loss_{}'.format(version), 'wb') as fo:
        pickle.dump({
            'loss':lossHist,
            'cost':costHist,
            'acc':accHist,
            'eta':etaHist
            },
            fo
        )
                   
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
    plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
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
    plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
    plt.show()
    
    # plot ACCURACY
    plt.plot(steps, [acc * 100 for acc in accHist], 'b', linewidth=2.5, alpha=1.0)
    plt.ylim(20,60)
    plt.xlim(0, steps[-1])
    plt.xlabel('Epoch')
    plt.ylabel('%', rotation=0, labelpad=20)
    plt.title('Testing accuracy')
    plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
    plt.show()