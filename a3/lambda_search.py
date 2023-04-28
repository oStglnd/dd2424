
import os
import json
import numpy as np

# get files
from train_network import trainNetwork

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a3\\results\\'

# set filename
fname = 'training_lambdaSearch'
fpath = results_path + fname

# set model params
batchNorm = True
alpha = 0.9
initialization = 'He'
sigma = 0
m = [50, 50]

# set training params
n_epochs = 50
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
ns      = 5 * 45000 // n_batch
n_cycles = 2

# set lambda values
l_min = -4
l_max = -2

# init dictionary for saving
saveDict = {
    'params':{
        'n_epochs':n_epochs,
        'n_batch':n_batch,
        'eta_min':eta_min,
        'eta_max':eta_max,
        'ns':ns,
        'n_cycles':n_cycles,
        'batchNorm':batchNorm,
        'alpha':alpha,
        'initialization':initialization,
        'sigma':sigma,
}}

# iterate over possible lambda values
for v in range(20):
    # generate lambda
    np.random.seed()
    l = l_min + (l_max - l_min) * np.random.rand()
    lambd = 10**l
    
    # get version
    # version = 'v' + str(v)
    version = 'lambdaSearch_' + str(v)
    
    print('\n TRAIN NETWORK ({}): cycles: {}, lambda: {:.3f}\n'.format(
        version,
        n_cycles,
        lambd
    ))
    
    # get training results
    lossHist, costHist, accHist = trainNetwork(
        n_epochs=n_epochs,
        n_batch=n_batch,
        eta_min=eta_min,
        eta_max=eta_max,
        ns=ns,
        n_cycles=n_cycles,
        lambd=lambd,
        m=m,
        batchNorm=batchNorm,
        alpha=alpha,
        initialization=initialization,
        sigma=sigma,
        version=version,
        plot=True
    )
    
    # save version-specific results in dictionary
    saveDict[version] = {
        'lambda':lambd,
        'lossHist':lossHist,
        'costHist':costHist,
        'accHist':accHist
    }
    
# dump results to JSON
with open(fpath, 'w') as fp:
    json.dump(saveDict, fp)