import torch
import torch.nn as nn
import torch.nn.functional as F

import lovmm.utils.utils as utils


class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.input_size=input_size
        self._make_layers()

    def _make_layers(self):
        self.fc1=nn.Sequential(nn.Flatten(),
                               nn.Linear(self.input_size, out_features=48),
                               
                                nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(48,48),
                               
                                nn.ReLU())
        self.fc3=nn.Sequential(nn.Linear(48,1))

    def forward(self, x):
        out = self.fc3(self.fc2(self.fc1(x)))
        return out

