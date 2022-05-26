import os
import sys
from math import ceil
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Function

sys.path.insert(0, os.getcwd())
from braindecode.models.layers import *

def init_kaiming(module):
    if hasattr(module, "weight"):
        nn.init.kaiming_normal_(module.weight.data)
    if hasattr(module, "bias"):
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class discriminator_DANN(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = int(1.0 * n_inputs)
        self.n_outputs = 1

        self.layers = nn.Sequential(
                GradientReversal(),
                nn.Linear(self.n_inputs, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_outputs)
        )

        init_kaiming(self.layers[1])
        init_kaiming(self.layers[3])
        init_kaiming(self.layers[5])

    def forward(self, x):
        x = self.layers(x)
        return x