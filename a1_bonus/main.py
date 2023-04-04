
import os
import numpy as np
import matplotlib.pyplot as plt

# get files
from model import linearClassifier
from misc import getCifar, getWeightImg, imgFlip, saveAsMat

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\a1_bonus\\plots\\'
model_path = home_path + '\\a1_bonus\\models\\'

###1
train_files = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]

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
X_train, X_val = X_train[:-1000], X_train[-1000:]
k_train, k_val = k_train[:-1000], k_train[-1000:]
Y_train, Y_val = Y_train[:-1000], Y_train[-1000:]

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
    activation='sigmoid',
    # activation='softmax',
    seed=400
)

# ###
# W_grads, b_grads = linearModel.computeGrads(
#     X=X_train[:100], 
#     Y=Y_train[:100], 
#     lambd=0.1
# )

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

# ### print gradient diff
# print(W_gradDiffMax, b_gradDiffMax)
# # raise SystemExit(0)

### train model
version     = 'v8'
lambd       = 0.1
eta         = 0.05
n_batch     = 100
n_epochs    = 50
n_decay     = 100
flip_p      = 0.2

# create lists for storing results
trainLoss, valLoss, trainCost, valCost, testAcc = [], [], [], [], []

# create list of idxs for shuffling
idxs = list(range(len(X_train)))

# print out parameter settings
print('PARAMETERS: \n\t Version: {}\n\t Lambda: {:.2f}\n\t Eta: {:.2f}\n\t N_batch: {}\n\t N_epochs: {}\n\t N_decay: {}\n\t P_flip: {:.2f}\n\n'.format(
    version,
    lambd,
    eta,
    n_batch,
    n_epochs,
    n_decay,
    flip_p
))

for epoch in range(n_epochs):
    # shuffle training examples
    np.random.shuffle(idxs)
    X_train, Y_train = X_train[idxs], Y_train[idxs]
    
    # # flip images
    if flip_p > 0:
        X_train = imgFlip(X_train, prob=flip_p)
    
    # decay learning rate
    if epoch % n_decay == 0:
        eta *= 0.1
    
    # iterate over batches
    for i in range(len(X_train) // n_batch):
        X_trainBatch = X_train[i*n_batch:(i+1)*n_batch]
        Y_trainBatch = Y_train[i*n_batch:(i+1)*n_batch]
        
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
saveAsMat(linearModel.W, model_path + 'model_{}_W'.format(version))
saveAsMat(linearModel.b, model_path + 'model_{}_b'.format(version))