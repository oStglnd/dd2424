
import os
import json
import numpy as np

# get files
from train_network import trainNetwork

versionDict = {
    'v7':{
        'optimizer':'',
        'm':[50, 50, 50, 20, 20, 10, 10],
        'cyclical': True,
        'decay': False
    },
    'v8':{
        'optimizer':'',
        'm':[100, 100, 100, 50, 50, 20, 20],
        'cyclical': True,
        'decay': False
    },
    'v9':{
        'optimizer':'adagrad',
        'm':[50, 50, 50, 20, 20, 10, 10],
        'cyclical': False,
        'decay': True
    },
    'v10':{
        'optimizer':'adagrad',
        'm':[100, 100, 100, 50, 50, 20, 20],
        'cyclical': False,
        'decay': True
    },
    'v11':{
        'optimizer':'adam',
        'm':[50, 50, 50, 20, 20, 10, 10],
        'cyclical': False,
        'decay': False
    },
    'v12':{
        'optimizer':'adam',
        'm':[100, 100, 100, 50, 50, 20, 20],
        'cyclical': False,
        'decay': False
    },
}


for version, params in versionDict.items():
    
    trainNetwork(
        version=version,
        m=params['m'],
        optimizer=params['optimizer'],
        cyclical=params['cyclical'],
        decay=params['decay']
    )