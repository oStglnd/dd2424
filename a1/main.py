
# get dependencies
import os
import pickle
import numpy as np

# get files
from model import linearClassifier
from misc import getCifar

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\plots\\a1\\'

###1
X_train, k_train, Y_train = getCifar(data_path + 'data_batch_1')
X_val, k_val, Y_val       = getCifar(data_path + 'data_batch_2')
X_test, k_test, Y_test    = getCifar(data_path + 'test_batch')

### 2
# whiten w. training data
mean_train = np.mean(X_train, axis=0)
std_train  = np.std(X_train, axis=0)

X_train = (X_train - mean_train) / std_train
X_test  = (X_test - mean_train) / std_train
X_val   = (X_val - mean_train) / std_train
