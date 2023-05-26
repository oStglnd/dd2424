
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
    'loss_v1',
    'loss_v2',
    'loss_v3',
    'loss_v4'
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
plt.plot([acc * 100 for acc in accDict['v1']], 'b', linewidth=1.5, alpha=1.0, label='vanilla')
plt.plot([acc * 100 for acc in accDict['v2']], 'm', linewidth=1.5, alpha=1.0, label='precise')
plt.plot([acc * 100 for acc in accDict['v3']], 'r', linewidth=1.5, alpha=1.0, label='adaptive')
plt.plot([acc * 100 for acc in accDict['v4']], 'g', linewidth=1.5, alpha=1.0, label='combined')


plt.ylim(30,60)
plt.xlim(0, len(accDict['v1'])-1)
plt.xlabel('Epoch')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy for variations on batchNorm')
plt.legend(loc='right')
plt.savefig(plot_path + 'acc_comp_bn.png', dpi=200)
plt.show()

# plot LOSS
plt.plot(lossDict['v1'], 'b', linewidth=1.5, alpha=1.0, label='vanilla')
plt.plot(lossDict['v2'], 'm', linewidth=1.5, alpha=1.0, label='precise')
plt.plot(lossDict['v3'], 'r', linewidth=1.5, alpha=1.0, label='adaptive')
plt.plot(lossDict['v4'], 'g', linewidth=1.5, alpha=1.0, label='combined')

plt.ylim(0.5,2.5)
plt.xlim(0, len(accDict['v1'])-1)
plt.xlabel('Epoch')
# plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Training loss for variations on batchNorm')
plt.legend(loc='right')
plt.savefig(plot_path + 'loss_comp_bn.png', dpi=200)
plt.show()
