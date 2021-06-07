import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MINE_SP(BaseNet):

    def __init__(self):
        super().__init__()

        rep = 50
        self.rep_dim = rep
        
        
        self.fc1 = nn.Linear(189, rep, bias=False)       
        self.fc2 = nn.Linear(rep, rep, bias=False)       
        self.fc3 = nn.Linear(rep, rep, bias=False)       
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        return x


