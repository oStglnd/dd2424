
import os
import numpy as np
import matplotlib.pyplot as plt

# get files
from model import neuralNetwork
from misc import getCifar, cyclicLearningRate

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\a1\\plots\\'
model_path = home_path + '\\a1\\models\\'

# get data
X_train, k_train, Y_train = getCifar(data_path, 'data_batch_1')
X_val, k_val, Y_val       = getCifar(data_path, 'data_batch_2')
X_test, k_test, Y_test    = getCifar(data_path, 'test_batch')

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
    m = 50,
    seed=1
)

# GRADS TEST
W1_grads, W2_grads, b1_grads, b2_grads = neuralNet.computeGrads(
    X=X_train[:1000], 
    Y=Y_train[:1000], 
    lambd=0
)

# W1_gradsNum, W2_gradsNum, b1_gradsNum, b2_gradsNum = neuralNet.computeGradsNumerical(
W1_gradsNum, W2_gradsNum, b1_gradsNum, b2_gradsNum = neuralNet.computeGradsNumerical(
    X=X_train[:1000], 
    Y=Y_train[:1000], 
    lambd=0,
    eps=1e-5
) 

# get max diffs
W1_gradDiffMax = np.max(np.abs(W1_grads[:10, :100] - W1_gradsNum[:10, :100]))
W2_gradDiffMax = np.max(np.abs(W2_grads[:10, :] - W2_gradsNum[:10, :]))
b1_gradDiffMax = np.max(np.abs(b1_grads[:10] - b1_gradsNum[:10]))
b2_gradDiffMax = np.max(np.abs(b2_grads[:10] - b2_gradsNum[:10]))
print('\nGradient check: \n\t max|W1 - W1_num| = {:.10f}\
                        \n\t max|W2 - W2_num| = {:.10f}\
                        \n\t max|b1 - b1_num| = {:.10f}\
                        \n\t max|b2 - b2_num| = {:.10f}\n'.format(
        W1_gradDiffMax, 
        W2_gradDiffMax, 
        b1_gradDiffMax, 
        b2_gradDiffMax
))

# test train
lossHist, costHist = [], []
for epoch in range(1,  200):
    if epoch % 50 == 0:
        print('EPOCH {} of gradient test training, \n loss: {:.3f}'.format(
            epoch,
            lossHist[-1]
        ))
    
    neuralNet.train(
        X_train[:100],
        Y_train[:100],
        lambd=0,
        eta=0.01
    )
    
    loss, cost = neuralNet.computeCost(
        X_train[:100], 
        Y_train[:100], 
        lambd=0
    )
    
    lossHist.append(loss)
    costHist.append(cost)
