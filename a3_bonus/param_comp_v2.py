
import json
import pickle
import os
import matplotlib.pyplot as plt

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a3_bonus\\results\\'
plot_path = home_path + '\\a3_bonus\\plots\\'

# set filename
fnames = [
    'loss_v7',
    'loss_v8',
    'loss_v9',
    'loss_v10',
    'loss_v11',
    'loss_v12'
]

lossDict = {}
costDict = {}
accDict = {}

for fname in fnames:
    
    fpath = results_path + fname
    
    with open(fpath, 'rb') as fp: 
        results = pickle.load(fp)

    lossDict[fname[-2:]] = results['loss']['train']
    costDict[fname[-2:]] = results['cost']['train']
    accDict[fname[-2:]] = results['acc']


# plot ACCURACY
plt.plot([acc * 100 for acc in accDict['v7']], 'b', linewidth=1.5, alpha=1.0, label='cyclical')
plt.plot([acc * 100 for acc in accDict['v8']], 'm', linewidth=1.5, alpha=1.0, label='cyclical, wide')
plt.plot([acc * 100 for acc in accDict['v9']], 'r', linewidth=1.5, alpha=1.0, label='adagrad')
plt.plot([acc * 100 for acc in accDict['10']], 'g', linewidth=1.5, alpha=1.0, label='adagrad, wide')
plt.plot([acc * 100 for acc in accDict['11']], 'tab:pink', linewidth=1.5, alpha=1.0, label='adam')
plt.plot([acc * 100 for acc in accDict['12']], 'tab:orange', linewidth=1.5, alpha=1.0, label='adam, wide')

plt.ylim(30,60)
plt.xlim(0, len(accDict['v7'])-1)
plt.xlabel('Epoch')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy for deep networks')
plt.legend(loc='lower right')
plt.savefig(plot_path + 'acc_comp_deep.png', dpi=200)
plt.show()

# # plot LOSS
plt.plot(lossDict['v7'], 'b', linewidth=1.5, alpha=1.0, label='cyclical')
plt.plot(lossDict['v8'], 'm', linewidth=1.5, alpha=1.0, label='cyclical, wide')
plt.plot(lossDict['v9'], 'r', linewidth=1.5, alpha=1.0, label='adagrad')
plt.plot(lossDict['10'], 'g', linewidth=1.5, alpha=1.0, label='adagrad, wide')
plt.plot(lossDict['11'], 'tab:pink', linewidth=1.5, alpha=1.0, label='adam')
plt.plot(lossDict['12'], 'tab:orange', linewidth=1.5, alpha=1.0, label='adam, wide')

# plt.ylim(30,60)
plt.xlim(0, len(accDict['v7'])-1)
plt.xlabel('Epoch')
# plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Training loss for deep networks')
plt.legend(loc='upper right')
plt.savefig(plot_path + 'loss_comp_deep.png', dpi=200)
plt.show()