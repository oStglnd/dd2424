
import os
import numpy as np
import matplotlib.pyplot as plt

# get files
from model import linearClassifier
from misc import getCifar, getWeightImg, saveAsMat

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\a1\\plots\\'
model_path = home_path + '\\a1\\models\\'

# get data
X_train, k_train, Y_train = getCifar(data_path, 'data_batch_1')
X_val, k_val, Y_val       = getCifar(data_path, 'data_batch_2')
X_test, k_test, Y_test    = getCifar(data_path, 'test_batch')

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
    d=X_train.shape[1],
    seed=400
)

# ###
# W_grads, b_grads = linearModel.computeGrads(
#     X=X_train, 
#     Y=Y_train, 
#     lambd=0.0
# )

# ###
# W_gradsNum, b_gradsNum = linearModel.computeGradsNumerical(
#     X=X_train, 
#     Y=Y_train, 
#     lambd=0.0,
#   eps = 1e-5
# )

# ### test gradient diff
# W_gradDiffMax = np.max(np.abs(W_grads - W_gradsNum))
# b_gradDiffMax = np.max(np.abs(b_grads - b_gradsNum))
# print(W_gradDiffMax, b_gradDiffMax)

### train model
version     = 'v2'
lambd       = 0.0
eta         = .001
n_batch     = 100
n_epochs    = 40

# create lists for storing results
trainLoss, valLoss, trainCost, valCost, testAcc = [], [], [], [], []

# create list of idxs for shuffling
idxs = list(range(len(X_train)))

for epoch in range(1, n_epochs+1):
    # shuffle training examples
    np.random.shuffle(idxs)
    X_train, Y_train = X_train[idxs], Y_train[idxs]
    
    # iterate over batches
    for i in range(len(X_train) // n_batch):
        X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
        Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
        
        # run training, GD update
        linearModel.train(
            X=X_trainBatch, 
            Y=Y_trainBatch, 
            lambd=lambd, 
            eta=eta
        )
    
    # compute cost and loss
    epochTrainLoss, epochTrainCost = linearModel.computeCost(
        X_train, 
        Y_train, 
        lambd=lambd
    )
    epochValLoss, epochValCost     = linearModel.computeCost(
        X_val, 
        Y_val, 
        lambd=lambd
    )
    
    # store vals
    trainLoss.append(epochTrainLoss)
    trainCost.append(epochTrainCost)
    valLoss.append(epochValLoss)
    valCost.append(epochValCost)
    
    # compute accuracy on test set
    testAcc.append(linearModel.computeAcc(X_test, k_test))
    
    # print info
    print(
        'EPOCH {} - trainingloss: {:.2f}, validationLoss: {:.2f}, testAcc: {:.4f}'\
        .format(epoch, trainLoss[-1], valLoss[-1], testAcc[-1])
    )
        
# plot COST function
plt.plot(trainCost, 'b', linewidth=1.5, alpha=1.0, label='Training')
plt.plot(valCost, 'r', linewidth=1.5, alpha=1.0, label='Validation')

plt.xlim(0, n_epochs)
plt.xlabel('Epoch')
plt.ylabel('Cost', rotation=0, labelpad=20)
plt.title('Cost per training epoch')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'cost_{}.png'.format(version), dpi=200)
plt.show()

# plot LOSS function
plt.plot(trainLoss, 'b', linewidth=1.5, alpha=1.0, label='Training')
plt.plot(valLoss, 'r', linewidth=1.5, alpha=1.0, label='Validation')

plt.xlim(0, n_epochs)
plt.xlabel('Epoch')
plt.ylabel('Loss', rotation=0, labelpad=20)
plt.title('Loss per training epoch')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'loss_{}.png'.format(version), dpi=200)
plt.show()

# plot ACCURACY
plt.plot([acc * 100 for acc in testAcc], 'm', linewidth=2.5, alpha=1.0)
plt.xlim(0, n_epochs)
plt.xlabel('Epoch')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy')
plt.savefig(plot_path + 'acc_{}.png'.format(version), dpi=200)
plt.show()

# plot WEIGHTS
wImgs = getWeightImg(linearModel.W)
_, axs = plt.subplots(1, 10)
for img, ax in zip(wImgs, axs.flatten()):
    ax.imshow(img)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.tick_params(left=False, bottom=False)

plt.savefig(plot_path + 'weights_{}.png'.format(version), bbox_inches='tight', dpi=500)
plt.show()

# save MODEL
# saveAsMat(linearModel.W, model_path + 'model_{}_W'.format(version))
# saveAsMat(linearModel.b, model_path + 'model_{}_b'.format(version))