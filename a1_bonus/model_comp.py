
import pickle
import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# get path
path = os.getcwd()
home_path = os.path.dirname(os.getcwd())
plot_path = home_path + '\\a1_bonus\\plots\\'

# set fnames
fnames = [
    'softmaxPreds',
    'sigmoidPreds',
    'truePreds'
]

# define dict
data = {}

# get files
for fname in fnames:
    with (open(path + '\\' + fname, 'rb')) as openfile:
        data[fname] = pickle.load(openfile)
        
# create dataframe
data = pd.DataFrame(data)

# get if TRUE or FALSE pred
data['softmaxPreds'] = data['softmaxPreds'] == data['truePreds']
data['sigmoidPreds'] = data['sigmoidPreds'] == data['truePreds']

# group data by class
dataGrouped = data.groupby('truePreds').mean()

# plot with seaborn
sns.set_style('white')

# plot softmaxPREDS
sns.barplot(
    x=dataGrouped.index, 
    y=dataGrouped['softmaxPreds'], 
    palette='Blues'
)

plt.ylim(0, 1.0)
plt.xlabel('Categories')
plt.ylabel('%')
plt.title('Correct share of class-specific predictions')
plt.savefig(plot_path + 'softmaxPreds.png', dpi=400)
plt.show()


# plot sigmoidPREDS
sns.barplot(
    x=dataGrouped.index, 
    y=dataGrouped['sigmoidPreds'], 
    palette='ch:.25'
)

plt.ylim(0, 1.0)
plt.xlabel('Categories')
plt.ylabel('%')
plt.title('Correct share of class-specific predictions')
plt.savefig(plot_path + 'sigmoidPreds.png', dpi=400)
plt.show()