
import json
import os

# get paths
home_path = os.path.dirname(os.getcwd())
results_path = home_path + '\\a2\\results\\'

# set filename
fname = 'training_v2'
fpath = results_path + fname

with open(fpath, 'r') as fp: 
    results = json.load(fp)
    

lambdas = []
accResults = []
for key, vals in results.items():
    if key != 'params':
        
        lambdas.append(vals['lambda'])
        accResults.append(vals['accHist'][-1])
        
        print('\t lambda: {:.5f}, accuracy: {:.4f}'.format(lambdas[-1], accResults[-1]))