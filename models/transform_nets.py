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


class TransformNet(nn.Module):
    """
    Network for inferring transforms applied on input data and feature matrices.
    """
    def __init__(self, size):
        self.transform_layer = nn.Linear(size, size, bias=False)

    def forward(self, X):
        return self.transform_layer(X)
