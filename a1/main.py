
import os
import numpy as np
import matplotlib.pyplot as plt

# get files
from model import linearClassifier
from misc import getCifar

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\plots\\a1\\'

###1
X_train, k_train, Y_train = getCifar(data_path + 'data_batch_1')
X_val, k_val, Y_val       = getCifar(data_path + 'data_batch_2')
X_test, k_test, Y_test    = getCifar(data_path + 'test_batch')

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
    d=X_train.shape[1]
)

###
W_grads, b_grads = linearModel.computeGrads(
    X=X_train[:100], 
    Y=Y_train[:100], 
    lambd=0.1
)

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


### train model
lambd       = 1.0
n_batch     = 100
eta         = 0.001
n_epochs    = 40

trainLoss, valLoss, testAcc = [], [], []
for epoch in range(n_epochs):
    for i in range(10000 // n_batch):
        X_trainBatch = X_train[i*n_batch:i*n_batch+n_batch]
        Y_trainBatch = Y_train[i*n_batch:i*n_batch+n_batch]
        
        linearModel.train(
            X=X_trainBatch, 
            Y=Y_trainBatch, 
            lambd=lambd, 
            eta=eta
        )
        
    trainLoss.append(linearModel.computeCost(X_train, Y_train, lambd=lambd))
    valLoss.append(linearModel.computeCost(X_val, Y_val, lambd=lambd))
    testAcc.append(linearModel.computeAcc(X_test, k_test))
    
    print(
        'EPOCH {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.2f}'\
        .format(epoch, trainLoss[-1], valLoss[-1], testAcc[-1]))
        
        
plt.plot(trainLoss, 'b', linewidth=1.5, alpha=1.0, label='Training loss')
plt.plot(valLoss, 'r', linewidth=1.5, alpha=1.0, label='Validation loss')

plt.xlim(0, n_epochs)
plt.xlabel('Epoch')
plt.ylabel('Loss', rotation=0, labelpad=15)
plt.legend(loc='upper right')
plt.show()

plt.plot(testAcc, 'm', linewidth=2.0, alpha=1.0)
plt.xlim(0, n_epochs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy', rotation=0, labelpad=30)
plt.title('Testing Accuracy')
plt.show()