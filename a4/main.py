
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

# define X, and Y, w. one-hot encoded representations
data = oneHotEncode(np.array([charToKey[char] for char in data]))
seqs = []
for i in range(len(data) - seq_length):
    seqs.append(data[i:i+seq_length])

X = seqs[1:]
Y = seqs[:-1]

# init networks
recurrentNet = recurrentNeuralNetwork(
    K=K,
    m=m,
    sigma=sigma,
    seed=200
)

lossList = []
smooth_loss = 0
e = 0
for i in range(200000):
    recurrentNet.train(X[e], Y[e], lambd=0, eta=0.1)
    loss, _ = recurrentNet.computeCost(X[e], Y[e], lambd=0)
    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                     
    lossList.append(smooth_loss)
    if i % 1000 == 0:
        print('Iteration {}, LOSS: {}'.format(i, smooth_loss))
        
    if i % 5000 == 0:
        sequence = recurrentNet.synthesizeText(
            x0=X[e+1][:1], 
            n=250
        )
        
        # convert to chars and print sequence
        sequence = ''.join([keyToChar[key] for key in sequence])
        print('\nGenerated sequence \n\t {}\n'.format(sequence))
        
    # update e
    e += seq_length