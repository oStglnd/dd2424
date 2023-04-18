
import os
import json

# get files
from train_network import trainNetwork

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a2_bonus\\results\\'

# set filename
fname = 'training_v3'
fpath = results_path + fname

with open(fpath, 'r') as fp: 
    results = json.load(fp)

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
    # 0.00015,
    # 0.0002,
    # 0.00025
]

mList = [
    50,
    75,
    100,
    125,
    150
]

pDropList = [
    0.2,
    0.3,
    0.4,
    0.5,
    0.0
]

# # init dictionary for saving
# saveDict = {
#     'params':{
#         'n_epochs':n_epochs,
#         'n_batch':n_batch,
#         'eta_min':eta_min,
#         'eta_max':eta_max,
#         'ns':ns,
#         'n_cycles':n_cycles,
#         #'m':m,
#         #'p_dropout':p_dropout,
#         'p_flip':p_flip,
#         'p_transl':p_transl,
#         'optimizer':optimizer
# }}

# iterate over possible lambda values
for v1, lambd in enumerate(lambdas):
    for v2, m in enumerate(mList):
        for v3, pDrop in enumerate(pDropList):
            # get version
            version = 'v' + str(v1) + str(v2) + str(v3)
            
            if version in results:
                continue
            
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
            # saveDict[version] = {
            #     'lambda':lambd,
            #     'm':m,
            #     'pDrop':pDrop,
            #     'lossHist':lossHist,
            #     'costHist':costHist,
            #     'accHist':accHist
            # }
            
            results[version] = {
                'lambda':lambd,
                'm':m,
                'pDrop':pDrop,
                'lossHist':lossHist,
                'costHist':costHist,
                'accHist':accHist
            }
            
            # dump results to JSON
            with open(fpath, 'w') as fp:
                json.dump(results, fp)