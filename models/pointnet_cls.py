"""
Simplified implementation of PointNet (Charles R. Q et al., CVPR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tnet_cls import TNetCls

import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("../utils")


class PointNetCls(torch.nn.Module):
    def __init__(self, input_dim=3, num_classes=40):
        """
        Constructor.

        Args:
        - input_dim: Int. Dimension of point cloud. Typically 3
        - num_classes: Int. Number of classes involved in a classification problem
        """

        super(PointNetCls, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_prop = 0.5

        # Conv layer for aggregating global features
        self.conv_1 = nn.Conv1d(self.input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 1024, 1)

        # Batch Norms for Conv layers
        self.bn_conv_1 = nn.BatchNorm1d(64)
        self.bn_conv_2 = nn.BatchNorm1d(64)
        self.bn_conv_3 = nn.BatchNorm1d(64)
        self.bn_conv_4 = nn.BatchNorm1d(128)
        self.bn_conv_5 = nn.BatchNorm1d(1024)

        # Fully Connected layers
        self.fc_1024_512 = nn.Linear(1024, 512)
        self.fc_512_256 = nn.Linear(512, 256)
        self.fc_256_out = nn.Linear(256, self.num_classes)

        # Batch Norms for FC layers
        self.bn_fc_1024_512 = nn.BatchNorm1d(512)
        self.bn_fc_512_256 = nn.BatchNorm1d(256)

        # Dropout for FC layers
        self.do_fc_1024_512 = nn.Dropout(p=self.dropout_prop)
        self.do_fc_512_256 = nn.Dropout(p=self.dropout_prop)

        # T-Nets
        self.tnet_1 = TNetCls(input_dim=3, affine_dim=3)
        self.tnet_2 = TNetCls(input_dim=64, affine_dim=64)

    def forward(self, x):
        """
        Forward propagation of PointNetCls

        Args:
        - x: A Tensor of shape (B, N, C)

        Returns: Unnormalized probability distribution of classes
        """

        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        # input transform
        t_mat_1 = self.tnet_1(x)
        x = torch.bmm(x, t_mat_1)
        x = x.transpose(1, 2)

        x = F.relu(self.bn_conv_1(self.conv_1(x)))
        x = F.relu(self.bn_conv_2(self.conv_2(x)))
        x = F.relu(self.bn_conv_3(self.conv_3(x)))

        # feature transform
        x = x.transpose(1, 2)
        t_mat_2 = self.tnet_2(x)
        x = torch.bmm(x, t_mat_2)
        x = x.transpose(1, 2)

        x = F.relu(self.bn_conv_4(self.conv_4(x)))
        x = F.relu(self.bn_conv_5(self.conv_5(x)))

        # max pooling (global feature aggregation)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.do_fc_1024_512(F.relu(self.bn_fc_1024_512(self.fc_1024_512(x))))
        x = self.do_fc_512_256(F.relu(self.bn_fc_512_256(self.fc_512_256(x))))
        x = self.fc_256_out(x)

        # x.shape = (B, num_classes)
        return x
