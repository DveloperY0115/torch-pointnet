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

# -------*-------
# hack tensor_repr for easier debugging
old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info
# -------*-------


class PointNet(torch.nn.Module):
    """
    PointNet class
    """
    def __init__(self, num_classes=9):
        """
        Constructor.
        """
        super(PointNet, self).__init__()

        # T-Nets
        self.t_net_input = TransformNet(size=3)
        self.t_net_feature = TransformNet(size=64)

        # MLPs -> Official implementation uses Conv Layers but following naive one suggested in the paper
        self.mlp_3_64 = nn.Linear(in_features=3, out_features=64)
        self.mlp_64_64_a = nn.Linear(in_features=64, out_features=64)
        self.mlp_64_64_b = nn.Linear(in_features=64, out_features=64)
        self.mlp_64_128 = nn.Linear(in_features=64, out_features=128)
        self.mlp_128_1024 = nn.Linear(in_features=128, out_features=1024)

        # Fully Connected Layers
        self.fc_1024_512 = nn.Linear(in_features=1024, out_features=512)
        self.fc_512_256 = nn.Linear(in_features=512, out_features=256)
        self.fc_256_k = nn.Linear(in_features=256, out_features=num_classes)

        # Batch normalization layers
        self.bn_64_a = nn.BatchNorm1d(num_features=64)
        self.bn_64_b = nn.BatchNorm1d(num_features=64)
        self.bn_128 = nn.BatchNorm1d(num_features=128)
        self.bn_1024 = nn.BatchNorm1d(num_features=1024)

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
