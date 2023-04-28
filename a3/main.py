
import os
import json
import numpy as np

# get files
from train_network import trainNetwork

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a3\\results\\'

# set filename
fname = 'testing_initSensitivity'
fpath = results_path + fname

# set model param
alpha = 0.9
m = [50, 50]
initialization = ''
lambd = 0.005

# set training params
n_epochs = 50
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
ns      = 5 * 45000 // n_batch
n_cycles = 3

# init dictionary for saving
saveDict = {
    'params':{
        'n_epochs':n_epochs,
        'n_batch':n_batch,
        'eta_min':eta_min,
        'eta_max':eta_max,
        'ns':ns,
        'n_cycles':n_cycles,
        'alpha':alpha,
        'initialization':initialization,
        'm':m,
}}


sigmaList = [
    1e-1,
    1e-3,
    1e-4
]

batchNormList = [
    True,
    False
]

# iterate over possible lambda values
for v1, sigma in enumerate(sigmaList):
    for v2, batchNorm in enumerate(batchNormList):
        
        # get version
        # version = 'v' + str(v)
        version = 'initSensitivity_' + str(v1) + str(v2)
        
        # if version in results:
        #     continue
        
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
            'batchNorm':batchNorm,
            'sigma':sigma,
            'lossHist':lossHist,
            'costHist':costHist,
            'accHist':accHist
        }
        
    # dump results to JSON
    with open(fpath, 'w') as fp:
        json.dump(saveDict, fp)