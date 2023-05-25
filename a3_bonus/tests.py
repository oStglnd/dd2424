
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from misc import getCifar
from model import neuralNetwork

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\a3\\plots\\'
model_path = home_path + '\\a3\\models\\'

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
    m = [50, 50, 30, 20],
    batchNorm=True,
    alpha=0.9,
    initialization='He',
    sigma=0,
    seed=1
)

# GRADS TEST
neuralNet.initBatchNorm(X_train[:100])

gradsListNum = neuralNet.computeGradsNumerical(
    X=X_train[:100], 
    Y=Y_train[:100], 
    lambd=0,
    eps=1e-5
) 

gradsList = neuralNet.computeGradsBatchNorm(
    X=X_train[:100], 
    Y=Y_train[:100], 
    lambd=0
)

print('\nGradient check:')
for idx, (grads, gradsNum) in enumerate(zip(gradsList, gradsListNum)):
    W_gradDiffMax = np.max(np.abs(grads['W'][:10, :10] - gradsNum['W'][:10, :10]))
    b_gradDiffMax = np.max(np.abs(grads['b'][:10] - gradsNum['b'][:10]))
    
    if idx < len(gradsList)-1:
        gamma_gradDiffMax = np.max(np.abs(grads['gamma'][:10, :10] - gradsNum['gamma'][:10, :10]))
        beta_gradDiffMax = np.max(np.abs(grads['beta'][:10] - gradsNum['beta'][:10]))
        
        print('\n Layer {}'.format(idx))
        print('\t max|W - W_num| = {:.10f}\
                \n\t max|b - b_num| = {:.10f}\
                \n\t max|gamma - gamma_num| = {:.10f}\
                \n\t max|beta - beta_num| = {:.10f}'.format(
                W_gradDiffMax, 
                b_gradDiffMax,
                gamma_gradDiffMax,
                beta_gradDiffMax
        ))
    
W_gradDiffMax = np.max(np.abs(grads['W'][:10, :10] - gradsNum['W'][:10, :10]))
b_gradDiffMax = np.max(np.abs(grads['b'][:10] - gradsNum['b'][:10]))
print('\n Layer {}'.format(idx))
print('\t max|W - W_num| = {:.10f}\
        \n\t max|b - b_num| = {:.10f}'.format(
        W_gradDiffMax, 
        b_gradDiffMax,
))


# test train
lossHist, costHist = [], []
for epoch in range(1,  500):
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

# plot results
plt.plot(lossHist, 'b', linewidth=1.5, alpha=1.0, label='Loss')
plt.plot(costHist, 'r--', linewidth=1.5, alpha=1.0, label='Cost')

plt.xlim(0, len(lossHist))
plt.xlabel('Step')
plt.ylabel('', rotation=0, labelpad=20)
plt.title('Training results for small subset')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'grad_test_9layer.png', dpi=200)
plt.show()