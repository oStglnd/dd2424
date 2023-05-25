
import json
import os
import matplotlib.pyplot as plt

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a3\\results\\'
plot_path = home_path + '\\a3\\plots\\'

# set filename
fname = 'testing_initSensitivity'
fpath = results_path + fname

with open(fpath, 'r') as fp: 
    results = json.load(fp)
    

lossLists = []
costLists = []
accLists = []
for key, vals in results.items():
    if key != 'params':
        
        lossLists.append(vals['lossHist'])
        costLists.append(vals['costHist'])
        accLists.append(vals['accHist'])
        # lambdas.append(vals['lambda'])
        # accResults.append(vals['accHist'][-1])
        
        # print('\t lambda: {:.5f}, accuracy: {:.4f}'.format(lambdas[-1], accResults[-1]))
        
# define steps for plot               
steps = [step * (2250 / 10) for step in range(len(accLists[0]))]

# plot ACCURACY f. sigma = 1.e-1
plt.plot(steps, [acc * 100 for acc in accLists[0]], 'b', linewidth=2.5, alpha=1.0, label='w. BatchNorm')
plt.plot(steps, [acc * 100 for acc in accLists[1]], 'm', linewidth=2.5, alpha=1.0, label='without')
plt.ylim(0,60)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy, $W$-intialization with $\mathcal{N}(0, 0.1)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'acc_comp_sigma1.png', dpi=200)
plt.show()

# plot ACCURACY f. sigma = 1.e-3
plt.plot(steps, [acc * 100 for acc in accLists[2]], 'b', linewidth=2.5, alpha=1.0, label='w. BatchNorm')
plt.plot(steps, [acc * 100 for acc in accLists[3]], 'm', linewidth=2.5, alpha=1.0, label='without')
plt.ylim(0,60)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy, $W$-intialization with $\mathcal{N}(0, 0.001)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'acc_comp_sigma2.png', dpi=200)
plt.show()

# plot ACCURACY f. sigma = 1.e-4
plt.plot(steps, [acc * 100 for acc in accLists[4]], 'b', linewidth=2.5, alpha=1.0, label='w. BatchNorm')
plt.plot(steps, [acc * 100 for acc in accLists[5]], 'm', linewidth=2.5, alpha=1.0, label='without')
plt.ylim(0,60)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('%', rotation=0, labelpad=20)
plt.title('Testing accuracy, $W$-intialization with $\mathcal{N}(0, 0.0001)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'acc_comp_sigma3.png', dpi=200)
plt.show()

# plot LOSS f. sigma = 1.e-1
plt.plot(steps, lossLists[0]['train'], 'g', linewidth=2.5, alpha=1.0, label='training, w. BatchNorm')
plt.plot(steps, lossLists[0]['val'], 'r', linewidth=2.5, alpha=1.0, label='validation, w. BatchNorm')
plt.plot(steps, lossLists[1]['train'], 'g--', linewidth=2.5, alpha=1.0, label='training, without')
plt.plot(steps, lossLists[1]['val'], 'r--', linewidth=2.5, alpha=1.0, label='validation, without')
plt.ylim(1.0, 2.5)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('Loss', rotation=0, labelpad=20)
plt.title('Loss,  $W$-intialization with $\mathcal{N}(0, 0.1)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'loss_comp_sigma1.png', dpi=200)
plt.show()


# plot LOSS f.  sigma = 1.e-3
plt.plot(steps, lossLists[2]['train'], 'g', linewidth=2.5, alpha=1.0, label='training, w. BatchNorm')
plt.plot(steps, lossLists[2]['val'], 'r', linewidth=2.5, alpha=1.0, label='validation, w. BatchNorm')
plt.plot(steps, lossLists[3]['train'], 'g--', linewidth=2.5, alpha=1.0, label='training, without')
plt.plot(steps, lossLists[3]['val'], 'r--', linewidth=2.5, alpha=1.0, label='validation, without')
plt.ylim(1.0, 2.5)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('Loss', rotation=0, labelpad=20)
plt.title('Loss,  $W$-intialization with $\mathcal{N}(0, 0.001)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'loss_comp_sigma2.png', dpi=200)
plt.show()

# plot LOSS f. sigma = 1.e-4
plt.plot(steps, lossLists[4]['train'], 'g', linewidth=2.5, alpha=1.0, label='training, w. BatchNorm')
plt.plot(steps, lossLists[4]['val'], 'r', linewidth=2.5, alpha=1.0, label='validation, w. BatchNorm')
plt.plot(steps, lossLists[5]['train'], 'g--', linewidth=2.5, alpha=1.0, label='training, without')
plt.plot(steps, lossLists[5]['val'], 'r--', linewidth=2.5, alpha=1.0, label='validation, without')
plt.ylim(1.0, 2.5)
plt.xlim(0, steps[-1])
plt.xlabel('Step')
plt.ylabel('Loss', rotation=0, labelpad=20)
plt.title('Loss,  $W$-intialization with $\mathcal{N}(0, 0.0001)$')
plt.legend(loc='right')
plt.savefig(plot_path + 'loss_comp_sigma3.png', dpi=200)
plt.show()