
import os
import json
import numpy as np
import string
import matplotlib.pyplot as plt

from misc import oneHotEncode

from model_v2 import batchRNN

# get paths
# home_path = os.path.dirname(os.getcwd())
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a4\\'
plot_path = home_path + '\\a4_bonus\\plots\\'
# results_path = home_path + '\\a4\\results\\'

# get text data
fname = 'goblet_book.txt'
fpath = data_path + fname

# read text file
with open(fpath, 'r') as fo:
    data = fo.readlines()


# split lines into words and words into chars
data = [char 
            for line in data
                for word in list(line)
                    for char in list(word)
]

# create word-key-word mapping
keyToChar = dict(enumerate(np.unique(data)))
charToKey = dict([(val, key) for key, val in keyToChar.items()])

# define params
K  = len(keyToChar)
m = 100
batch_size = 8
sigma = 0.01
seq_length = 25

# define X, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])

# stack X into data matrix
X = np.stack(X).transpose(0, 2, 1).astype('int8')

# partition X into blocks
n_blocks = 100
block_size = len(X) // n_blocks
X_blocks = [X[block_size*i:block_size*(i+1)] for i in range(block_size-1)]



# init networks
recurrentNet = batchRNN(
    K=K,
    m=m,
    batch_size=batch_size,
    sigma=sigma,
    seed=2
)

# save best weights
weights_best = recurrentNet.weights.copy()

epoch_n = 0
print ('\n------EPOCH {}--------\n'.format(epoch_n))

lossHist = []
loss_smooth, _ = recurrentNet.computeCost(X[:batch_size], X[1:batch_size+1], lambd=0)
loss_best = loss_smooth

n = len(X)
e = 0
for i in range(100000):
    recurrentNet.train(X[e:e+batch_size], X[e+1:e+batch_size+1], lambd=0, eta=0.1)
    loss, _ = recurrentNet.computeCost(X[e:e+batch_size], X[e+1:e+batch_size+1], lambd=0)
    
    loss_smooth = 0.999 * loss_smooth + 0.001 * loss
    if loss_smooth < loss_best:
        weights_best = recurrentNet.weights.copy()
        loss_best = loss_smooth

    if (i % 1000 == 0) and i > 0:
        lossHist.append(loss_smooth)
        print('Iteration {}, LOSS: {}'.format(i, loss_smooth))
        
    if i % 10000 == 0:
        sequence = recurrentNet.synthesizeText(
            x0=X[e+batch_size+1].T[:1], 
            n=250
        )
        
        # convert to chars and print sequence
        sequence = ''.join([keyToChar[key] for key in sequence])
        print('\nGenerated sequence \n\n {}\n'.format(sequence))
        
    # update e
    if e < (n - seq_length):
        e = min(np.random.randint(len(X)), len(X) - batch_size - 1)
        
    else:
        e = 0
        recurrentNet.hprev = np.zeros(shape=(m, batch_size))
        recurrentNet.cprev = np.zeros(shape=(m, batch_size))
        
        epoch_n += 1
        print ('\n------EPOCH {}--------\n'.format(epoch_n))
        
        if epoch_n >= 3:
            break
            
# plot results
steps = [step * 10 for step in range(len(lossHist))]
plt.plot(steps, lossHist, 'r', linewidth=1.5, alpha=1.0, label='Loss')
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('', rotation=0, labelpad=20)
plt.title('Smooth loss for $4$ epochs')
# plt.legend(loc='upper right')
plt.savefig(plot_path + 'rnn_loss.png', dpi=200)
plt.show()

recurrentNet.weights = weights_best
sequence = recurrentNet.synthesizeText(
    x0=X[e+1][:1], 
    n=1000
)

# convert to chars and print sequence
sequence = ''.join([keyToChar[key] for key in sequence])
print('\nGenerated sequence \n\n {}\n'.format(sequence))