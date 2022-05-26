from math import ceil
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Function

class net_wrapper(nn.Module):
    def __init__(self, net_cls):
        super().__init__()
        self.net_cls = net_cls

    def forward(self, x):

        preds = self.net_cls(x)

        return preds
