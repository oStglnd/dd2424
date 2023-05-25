
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

plotData = pd.DataFrame(valLossDict)

sns.relplot(
    data=plotData[['v2', 'v4', 'v6']],
    kind='line',
    dashes=False,
    markers=False,
    palette='Blues',
    linewidth=1.5
)

plt.xlim(0, 45000)
plt.ylim(2.0, 4.0)
plt.xlabel('Steps')
plt.ylabel('Loss', rotation=0, labelpad=15)
plt.title('Validation loss for Adam')
plt.savefig(plot_path + 'loss_comp_adam.png', dpi=200, bbox_inches='tight')
plt.show()

sns.relplot(
    data=plotData[['v1', 'v3', 'v5']],
    kind='line',
    dashes=False,
    markers=False,
    palette='crest',
    linewidth=1.5,
    legend='full'
)

plt.xlim(0, 45000)
plt.ylim(2.0, 4.0)
plt.xlabel('Steps')
plt.ylabel('Loss', rotation=0, labelpad=15)
plt.title('Validation loss for AdaGrad')
plt.savefig(plot_path + 'loss_comp_adagrad.png', dpi=200, bbox_inches='tight')
plt.show()

# pDropAcc = plotData.groupby('pDrop').mean()['acc'] * 100
# mAcc = plotData.groupby('m').mean()['acc'] * 100

# # plot dropout accuracy
# sns.barplot(
#     x=pDropAcc.index, 
#     y=pDropAcc.values, 
#     palette='magma'
# )

# plt.ylim(45, 55)
# plt.xlabel('Dropout rate, $p_{drop}$')
# plt.ylabel('%')
# plt.title('Mean accuracy per dropout rates')
# plt.savefig(plot_path + 'compDropout.png', dpi=400)
# plt.show()

# # plot m accuracy
# sns.barplot(
#     x=mAcc.index.astype('int64'), 
#     y=mAcc.values, 
#     palette='viridis'
# )

# plt.ylim(45, 55)
# plt.xlabel('Number of hidden nodes, $m$')
# plt.ylabel('%')
# plt.title('Mean accuracy per layer width')
# plt.savefig(plot_path + 'compLayerM.png', dpi=400)
# plt.show()