
import json
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a4_bonus\\results\\'
plot_path = home_path + '\\a4_bonus\\plots\\'

trainLossDict = {}
valLossDict = {}
fnames = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
for fname in fnames:
    fpath = results_path + 'loss_{}'.format(fname)
    
    with open(fpath, 'rb') as fp: 
        results = pickle.load(fp)
    
    if (fname == 'v5') or (fname == 'v6'):
        trainLossDict[fname] = np.array(results['train']) / 50
        valLossDict[fname] = np.array(results['val']) / 50
    else:
        trainLossDict[fname] = np.array(results['train'])[:41548] / 25
        valLossDict[fname] = np.array(results['val'])[:41548] / 25


# plot LOSS
plt.plot(trainLossDict['v3'], 'b', linewidth=1.5, alpha=1.0, label='AdaGrad, $m=100$, $sl=25$')
plt.plot(trainLossDict['v5'], 'm', linewidth=1.5, alpha=1.0, label='AdaGrad, $m=200$, $sl=50$')
plt.plot(trainLossDict['v4'], 'r', linewidth=1.5, alpha=1.0, label='Adam, $m=100$, $sl=25$')
plt.plot(trainLossDict['v6'], 'g', linewidth=1.5, alpha=1.0, label='Adam, $m=200$, $sl=50$')

# plt.ylim(0.5,2.5)
plt.xlim(0, len(trainLossDict['v1'])-1)
plt.xlabel('Epoch')
# plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Training loss for RNN variations')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'loss_train_comp_rnn.png', dpi=200)
plt.show()

# plot LOSS
plt.plot(valLossDict['v3'], 'b', linewidth=1.5, alpha=1.0, label='AdaGrad, $m=100$, $sl=25$')
plt.plot(valLossDict['v5'], 'm', linewidth=1.5, alpha=1.0, label='AdaGrad, $m=200$, $sl=50$')
plt.plot(valLossDict['v4'], 'r', linewidth=1.5, alpha=1.0, label='Adam, $m=100$, $sl=25$')
plt.plot(valLossDict['v6'], 'g', linewidth=1.5, alpha=1.0, label='Adam, $m=200$, $sl=50$')

# plt.ylim(0.5,2.5)
plt.xlim(0, len(valLossDict['v1'])-1)
plt.xlabel('Epoch')
# plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Validation loss for RNN variations')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'loss_val_comp_rnn.png', dpi=200)
plt.show()