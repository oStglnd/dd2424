
import os
import json
import numpy as np

from misc import oneHotEncode
from model import recurrentNeuralNetwork

# get paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a4\\'
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
                    for char in list(word)]

# create word-key-word mapping
keyToChar = dict(enumerate(np.unique(data)))
charToKey = dict([(val, key) for key, val in keyToChar.items()])

# define params
K  = len(keyToChar)
m = 100
sigma = 0.01
seq_length = 25

# define X, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
X = []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])

# init networks
recurrentNet = recurrentNeuralNetwork(
    K=K,
    m=m,
    sigma=sigma,
    seed=2
)

lossList = []
smooth_loss = 0
n = len(X)
e = 0
for i in range(300000):
    recurrentNet.train(X[e], X[e+1], lambd=0, eta=0.1)
    loss, _ = recurrentNet.computeCost(X[e], X[e+1], lambd=0)
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

    if (i % 100 == 0) and i > 0:
        lossList.append(smooth_loss)
        print('Iteration {}, LOSS: {}'.format(i, smooth_loss))
        
    if i % 1000 == 0:
        sequence = recurrentNet.synthesizeText(
            x0=X[e+1][:1], 
            n=250
        )
        
        # convert to chars and print sequence
        sequence = ''.join([keyToChar[key] for key in sequence])
        print('\nGenerated sequence \n\t {}\n'.format(sequence))
        
    # update e
    if e < (n - seq_length - 2):
        e += seq_length
    else:
        e = 0
        recurrentNet.hprev = np.zeros(shape=(m, 1))