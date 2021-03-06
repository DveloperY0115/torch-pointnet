import torch
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join('../utils'))


class PointNet_Basic(torch.nn.Module):
    """
    PointNet class
    """
    def __init__(self, D_in, H, D_out):
        """
        Constructor.

        :param D_in:
        :param H:
        :param D_out:
        """
        pass

    def forward(self, X):
        """
        Forward propagation.
        :param X: Input data
        :return: Pr
        """