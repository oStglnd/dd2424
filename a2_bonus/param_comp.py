
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a2_bonus\\results\\'
plot_path = home_path + '\\a2_bonus\\plots\\'

# set filename
fname = 'training_v3'
fpath = results_path + fname

with open(fpath, 'r') as fp: 
    results = json.load(fp)
    

dataDict = {}
# accResults = []
for key, vals in results.items():
    if key != 'params':
        
        dataDict[key] = {
            'lambda':vals['lambda'],
            'm':vals['m'],
            'pDrop':vals['pDrop'],
            'acc':max(vals['accHist'])
        }
        
        print('{:.5f} & {} & {:.2f} & {:.4f} \\\\'.format(
            vals['lambda'],
            vals['m'],
            vals['pDrop'],
            max(vals['accHist'])*100
        ))

# make DF
plotData = pd.DataFrame(dataDict).T

# get plotVals
pDropAcc = plotData.groupby('pDrop').mean()['acc'] * 100
mAcc = plotData.groupby('m').mean()['acc'] * 100

# plot dropout accuracy
sns.barplot(
    x=pDropAcc.index, 
    y=pDropAcc.values, 
    palette='magma'
)

plt.ylim(45, 55)
plt.xlabel('Dropout rate, $p_{drop}$')
plt.ylabel('%')
plt.title('Mean accuracy per dropout rates')
plt.savefig(plot_path + 'compDropout.png', dpi=400)
plt.show()

# plot m accuracy
sns.barplot(
    x=mAcc.index.astype('int64'), 
    y=mAcc.values, 
    palette='viridis'
)

plt.ylim(45, 55)
plt.xlabel('Number of hidden nodes, $m$')
plt.ylabel('%')
plt.title('Mean accuracy per layer width')
plt.savefig(plot_path + 'compLayerM.png', dpi=400)
plt.show()