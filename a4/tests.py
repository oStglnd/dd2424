
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

gradsListNum = recurrentNet.computeGradsNumerical(
    X[2], 
    Y[2], 
    lambd=0, 
    eps=1e-5
)

gradsList = recurrentNet.computeGrads(
    X[2], 
    Y[2], 
    lambd=0
)

print('\nGradient check:')
for key, grads in gradsList.items():
    W_gradDiffMax = np.max(np.abs(grads[:50, :50] - gradsListNum[key][:50, :50]))
    print('\t max|W - W_num| = {:.10f}'.format(W_gradDiffMax))