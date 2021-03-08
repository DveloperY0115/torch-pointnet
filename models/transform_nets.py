import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join('../utils'))


# TODO: Implement T-Net following the exact specification
class TransformNet(nn.Module):
    """
    Network for inferring transforms applied on input data and feature matrices.
    """
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.transform_layer = nn.Linear(in_features=size, out_features=size, bias=False)

    def forward(self, x):
        return self.transform_layer(x)
