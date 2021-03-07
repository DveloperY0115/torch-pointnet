import torch
import torch.nn as nn
import torch.nn.functional as F
from transform_nets import TransformNet
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join('../utils'))


class PointNet(torch.nn.Module):
    """
    PointNet class
    """
    def __init__(self, num_classes=9):
        """
        Constructor.
        """
        super(PointNet, self).__init__()

        # T-Net for input data
        self.t_net_input = TransformNet(size=3)
        self.t_net_feature = TransformNet(size=64)

        # MLPs for feature extraction
        # Consists of 1-D convolution layers
        self.mlp_64 = nn.Conv1d(3, 64, 1)
        self.mlp_128 = nn.Conv1d(64, 128, 1)
        self.mlp_1024 = nn.Conv1d(128, 1024, 1)

        # Fully Connected layers for classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)    # 9 Classes

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(64)   # batch norm after mlp_64
        self.bn2 = nn.BatchNorm1d(128)    # batch norm after mlp_128
        self.bn3 = nn.BatchNorm1d(1024)    # batch norm after mlp_1024
        self.bn4 = nn.BatchNorm1d(512)    # batch norm after fc1
        self.bn5 = nn.BatchNorm1d(256)    # batch norm after fc2

    def forward(self, X):




    def forward(self, X):
        """
        Forward propagation.
        :param X: Input data
        :return: Pr
        """