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

from transform_nets import TransformNet

class PointNetCls(torch.nn.Module):
    """
    PointNet class
    """
    def __init__(self, input_dim=3, num_classes=9):
        """
        Constructor.
        """
        super(PointNet, self).__init__()
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


    def forward(self, x):
        """
        Forward propagation.
        :param x: Input data
        :return: Probability mass function containing probability of each class
        """
        batch_size = x.size()[0]
        num_points = x.size()[1]

        x = self.t_net_input(x)    # (B, N, 3) -> (B, N, 3)
        x = self.mlp_3_64(x)    # (B, N, 3) -> (B, N, 64)

        # Batch normalization after 3 -> 64
        x = x.view(batch_size, -1, num_points)
        x = self.bn_64_a(x)
        x = x.view(batch_size, num_points, -1)

        x = self.mlp_64_64_a(x)    # (B, N, 64) -> (B, N, 64)

        # Batch normalization after 64 -> 64
        x = x.view(batch_size, -1, num_points)
        x = self.bn_64_b(x)
        x = x.view(batch_size, num_points, -1)

        x = self.t_net_feature(x)    # (B, N, 64) -> (B, N, 64)

        x = self.mlp_64_128(x)    # (B, N, 64) -> (B, N, 128)

        # Batch normalization after 64 -> 128
        x = x.view(batch_size, -1, num_points)
        x = self.bn_128(x)
        x = x.view(batch_size, num_points, -1)

        x = self.mlp_128_1024(x)    # (B, N, 128) -> (B, N, 1024)

        # Batch normalization after 128 -> 1024
        x = x.view(batch_size, -1, num_points)
        x = self.bn_1024(x)
        x = x.view(batch_size, num_points, -1)

        # Apply max pooling
        x, _ = torch.max(x, dim=1)    # (B, N, 1024) -> (B, 1024)

        x = self.fc_1024_512(x)
        x = self.fc_512_256(x)
        x = self.fc_256_k(x)

        return x
