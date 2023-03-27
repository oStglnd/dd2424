
# get dependencies
import os
import pickle
import numpy as np

# define paths
home_path = os.path.dirname(os.getcwd())
data_path = home_path + '\\data\\a1\\'
plot_path = home_path + '\\plots\\a1\\'

### 1
# get data
data_dict = {}
label_dict = {}
for file in ['1', '2', 'test']:
    fpath = data_path + 'data_batch_{}'.format(file)
    with open(fpath, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        
        # data_dict[file] = np.reshape(
        #     a=batch[b'data'],
        #     newshape=(10000,32,32,3)
        # )
        data_dict[file] = batch[b'data']
        label_dict[file] = batch[b'labels']
        
# get X data
X_train = np.array(data_dict['1'])
X_val   = np.array(data_dict['2'])
X_test  = np.array(data_dict['test'])

# get label data
k_train = np.array(label_dict['1'])
k_val   = np.array(label_dict['2'])
k_test  = np.array(label_dict['test'])

# get Y data
def oneHotEncode(k):
    return np.array([[
               1 if idx == label else 0 for idx in range(10)]
                for label in k_test]
            )

Y_train = oneHotEncode(k_train)
Y_val   = oneHotEncode(k_val)
Y_test  = oneHotEncode(k_test)

# delete dicts and batch
del batch, data_dict, label_dict

### 2
# whiten w. training data
mean_train = np.mean(X_train, axis=0)
std_train  = np.std(X_train, axis=0)

X_train = (X_train - mean_train) / std_train
X_test  = (X_test - mean_train) / std_train
X_val   = (X_val - mean_train) / std_train


### 3, 4, 5, 6, 7 (implemented as a class)
class linearClassifier:
    def __init__(self, K, d):
        self.K = K
        self.d = d
        
        # init weights
        W = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, d)
        )
        
        # init bias
        b = np.random.normal(
            loc=0, 
            scale=0.1, 
            size=(self.K, 1)
        )

    def evaluate(self, X):
        
        return 0
    
    def computeCost(self, X, Y, lambd):
        
        return 0
    
    def computeAcc(self, X, y):
        
        return 0
    
    def computeGrads(self, X, Y, P, lambd):
        
        return 0


# initialize model params
### 4
