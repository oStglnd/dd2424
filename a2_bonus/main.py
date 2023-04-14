
import os
import json
import numpy as np

# get files
from train_network import trainNetwork

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a2\\results\\'

# set filename
fname = 'training_v2'
fpath = results_path + fname

# set params
n_epochs = 50
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
ns      = 2 * 45000 // n_batch
n_cycles = 2
p_flip = 0.2
p_transl = 0.05

# specify network
m = 100
p_dropout = 0.2
optimizer = 'Adam'
# optimizer = ''

# set lambda values
l_min = -5
l_max = -1

# init dictionary for saving
saveDict = {
    'params':{
        'n_epochs':n_epochs,
        'n_batch':n_batch,
        'eta_min':eta_min,
        'eta_max':eta_max,
        'ns':ns,
        'n_cycles':n_cycles,
        'm':m,
        'p_dropout':p_dropout,
        'p_flip':p_flip,
        'p_transl':p_transl,
        'optimizer':optimizer
}}

# iterate over possible lambda values
for v in range(20):
    # generate lambda
    np.random.seed()
    l = l_min + (l_max - l_min) * np.random.rand()
    lambd = 10**l
    
    # get version
    version = 'v' + str(v)
    
    print('\n TRAIN NETWORK ({}): cycles: {}, lambda: {:.5f}\n'.format(
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
        m=m,
        p_dropout=p_dropout,
        p_flip=p_flip,
        p_transl=p_transl,
        optimizer=optimizer,
        lambd=lambd,
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