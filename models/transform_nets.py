import torch
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join('../utils'))


class TransformNet(torch.nn.Module):
    """
    Network for inferring transforms applied on input data and feature matrices.
    """
    def __init__(self, D_in, H, D_out):
        pass

    def forward(self, X):
        pass