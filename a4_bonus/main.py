
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from misc import oneHotEncode

from model import RNN
from model_v2 import batchRNN

# get paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a4\\'
plot_path = home_path + '\\a4_bonus\\plots\\'
models_path = home_path + '\\a4_bonus\\models\\'
results_path = home_path + '\\a4_bonus\\results\\'

# get text data
fname = 'goblet_book.txt'
fpath = data_path + fname

# read text file
with open(fpath, 'r') as fo:
    data = fo.readlines()

# spec certain chars to remove
removeList = ['0', '1', '2', '3', '4', '6', '7', '9', '}', '¢', '¼', 'Ã', 'â', '€']

# split lines into words and words into chars
data = [char 
            for line in data
                for word in list(line)
                    for char in list(word)
                        if char not in removeList
]

# create word-key-word mapping
keyToChar = dict(enumerate(np.unique(data)))
charToKey = dict([(val, key) for key, val in keyToChar.items()])

# define data specs
seq_length = 50

# define X, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length].astype('int8'))

# partition X into blocks, get validation block
n_blocks = 100
block_size = len(X) // n_blocks
X_blocks = [X[block_size*i:block_size*(i+1)] for i in range(n_blocks-1)]

# # choose random blocks for validation
# idxs = np.random.choice(range(n_blocks), size=10)
# val_block = []
# for count, idx in enumerate(idxs):
#     val_block.append(X_blocks.pop(idx-count))
X_blocks, val_block = X_blocks[:-5], X_blocks[-5:]
val_block = np.vstack(val_block)

# init networks
version = 'v7'
K  = len(keyToChar)
m = 200
sigma = 0.01
initialization = 'He'
optimizer = 'adam'
eta = 0.001
diverse = True

recurrentNet = RNN(
    K=K,
    m=m,
    sigma=sigma,
    initialization=initialization,
    optimizer=optimizer,
    seed=2
)

# save best weights
weights_best = recurrentNet.weights.copy()

# init loss metrics
trainLossHist = []
valLossHist = []
# loss_smooth, _ = recurrentNet.computeCost(X[0], X[1], lambd=0)
loss_smooth = seq_length * 4
loss_best = loss_smooth

valLoss_smooth = seq_length * 4

# init global counters
t = 0
block_n = 0
epoch_n = 0

print ('\n------EPOCH {}--------\n'.format(epoch_n))
while epoch_n < 5:
    if diverse:
        np.random.shuffle(X_blocks)
    
    for block in X_blocks:
        # reset hPrev
        if diverse:
            recurrentNet.hprev = np.zeros(shape=(m, 1))
        
        # reset counter
        e = 0
        while e < (block_size - seq_length):
            # train on sequence
            recurrentNet.train(block[e], block[e+1], block_n, eta=eta)
            loss, _ = recurrentNet.computeCost(block[e], block[e+1], lambd=0)
            loss_smooth = 0.999 * loss_smooth + 0.001 * loss
            
            # get random validation example
            valIdx = np.random.randint(len(val_block))-1
            valLoss, _ = recurrentNet.computeCost(val_block[valIdx], val_block[valIdx+1], lambd=0)
            valLoss_smooth = 0.999 * valLoss_smooth + 0.001 * valLoss
            
            # save loss metrics
            trainLossHist.append(loss_smooth)
            valLossHist.append(valLoss_smooth)
            
            # save weights if best loss
            if loss_smooth < loss_best:
                weights_best = recurrentNet.weights.copy()
                loss_best = loss_smooth
                
            # print generated sequence
            if t % 10000 == 0:
                sequence = recurrentNet.synthesizeText(
                    x0=block[e+1][:1], 
                    n=250,
                    T=1.0,
                    theta=0.9
                )
                
                # convert to chars and print sequence
                sequence = ''.join([keyToChar[key] for key in sequence])
                print('\nGenerated sequence \n\n {}\n'.format(sequence))
            
            # update e, t
            t += 1
            e += seq_length
    
        
        # compute validation loss
        # reset hPrev
        if diverse:
            recurrentNet.hprev = np.zeros(shape=(m, 1))
    
        # print metrics
        print('Step {}, block: {}, train LOSS: {:.2f}, val LOSS: {:.2f}'.format(
            t, 
            block_n,
            loss_smooth,
            valLoss_smooth
        ))
        
        block_n += 1
        
    # update EPOCH
    epoch_n += 1
    print('\n------EPOCH {}--------\n'.format(epoch_n))
        
    # reset hPrev
    recurrentNet.hprev = np.zeros(shape=(m, 1))
 
# trainLossHist = [val / seq_length for val in trainLossHist]
# valLossHist = [val / seq_length for val in valLossHist] 
          
# plot results
steps = [step * block_size for step in range(len(trainLossHist))]
plt.plot(steps, trainLossHist, 'r', linewidth=1.5, alpha=1.0, label='Training')
plt.plot(steps, valLossHist, 'g', linewidth=1.5, alpha=1.0, label='Validation')
plt.xlim(0, steps[-1])
# plt.ylim(1.5,4.0)
plt.ylim(30,)
plt.xlabel('Steps')
plt.ylabel('', rotation=0, labelpad=20)
plt.title('Smooth loss for $5$ epochs')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'rnn_loss_{}.png'.format(version), dpi=200)
plt.show()

# save model
with open(models_path + 'model_{}'.format(version), 'wb') as fo:
    pickle.dump(recurrentNet.weights, fo)

# save results
with open(results_path + 'loss_{}'.format(version), 'wb') as fo:
    pickle.dump({
        'train':trainLossHist,
        'val':valLossHist
        },
        fo
    )

# recurrentNet.weights = weights_best
recurrentNet.hprev = np.zeros(shape=(m, 1))
sequence = recurrentNet.synthesizeText(
    x0=X_blocks[0][0][:1], 
    n=300,
    T=0.5,
    theta=0.9
)

# convert to chars and print sequence
sequence = ''.join([keyToChar[key] for key in sequence])
print('\nGenerated sequence \n\n {}\n'.format(sequence))