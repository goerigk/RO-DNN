from base.torchvision_dataset import TorchvisionDataset

import math
import torch
import pandas
import numpy as np

class MINE_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):


        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = [0]
        self.outlier_classes = [1]


        self.train_set = []
        trdat = np.array(pandas.read_csv(root, header=None))

        for i in range(0,len(trdat)):
            self.train_set.append((torch.Tensor(trdat[i][:].astype(np.float)),0,i))
        
        self.test_set = []


    def __getitem__(self, index):
        if self.train:
            return self.train_set[index][0], self.train_set[index][1], index
        else:
            return self.test_set[index][0], self.test_set[index][1], index

