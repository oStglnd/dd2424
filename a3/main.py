
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from misc import getCifar, cyclicLearningRate
from model import neuralNetwork

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\a3\\plots\\'
model_path = home_path + '\\a3\\models\\'

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

# INIT params
K = 10      # init n. of classes
d = 3072    # init dimensions
m = [50, 30, 20, 20, 10, 10, 10, 10] # init units
alpha = 0.9 # init batchNorm param
seed = 200      # init seed

# generate model
neuralNet = neuralNetwork(
    K = K,
    d = d,
    m = m,
    alpha = alpha,
    seed=seed
)

# # set params
# n_epochs = 50
# n_batch = 100
# eta_min = 1e-5
# eta_max = 1e-1
# ns      = 5 * 45000 // n_batch
# n_cycles = 2
# lambd = 0.005

# # init lists/dicts
# etaHist, accHist = [], []
# lossHist, costHist = {'train':[], 'val':[]}, {'train':[], 'val':[]}

# # create list of idxs for shuffling
# idxs = list(range(len(X_train)))

# # create timestep
# t = 0

# for epoch in range(1, n_epochs+1):
#     # shuffle training examples
#     np.random.shuffle(idxs)
#     X_train, Y_train, k_train = X_train[idxs], Y_train[idxs], k_train[idxs]
    
#     # iterate over batches
#     for i in range(len(X_train) // n_batch):
#         X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
#         Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
        
#         # update eta
#         eta = cyclicLearningRate(
#             etaMin=eta_min, 
#             etaMax=eta_max, 
#             stepSize=ns, 
#             timeStep=t
#         )        
        
#         # run training, GD update
#         neuralNet.train(
#             X=X_trainBatch, 
#             Y=Y_trainBatch, 
#             lambd=lambd, 
#             eta=eta
#         )
        
#         # append to list of eta and update time step
#         etaHist.append(eta)
#         t += 1
        
#         # if some number of cycles, break
#         if t >= n_cycles * 2 * ns:
#             break
    
#         # add loss, cost info, 4 times per cycle
#         if t % (ns / 10) == 0:
#             trainLoss, trainCost = neuralNet.computeCost(
#                 X_train, 
#                 Y_train, 
#                 lambd=lambd
#             )
            
#             valLoss, valCost = neuralNet.computeCost(
#                 X_val, 
#                 Y_val, 
#                 lambd=lambd
#             )
            
#             # get acc
#             acc = neuralNet.computeAcc(X_test, k_test)
            
#             # save info
#             lossHist['train'].append(trainLoss)
#             lossHist['val'].append(valLoss)
#             costHist['train'].append(trainCost)
#             costHist['val'].append(valCost)
#             accHist.append(acc)
        
#             # print info
#             print(
#                 '\t STEP {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.4f}'\
#                 .format(t, lossHist['train'][-1], costHist['train'][-1], accHist[-1])
#             )