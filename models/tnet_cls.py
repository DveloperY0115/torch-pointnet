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


class TNet_Cls(nn.Module):

    def __init__(self, input_dim=3, affine_dim=3):
        """
        Constructor.

        Args:
        - input_dim: Int. Dimension of point cloud. Typically 3
        """

        super(TNet_Cls, self).__init__()
        self.input_dim = input_dim
        self.affine_dim = affine_dim
        self.dropout_prop = 0.5

        # Conv layer
        self.conv_1 = nn.Conv1d(self.input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024 ,1)

        # Batch Norms for Conv layers
        self.bn_conv_1 = nn.BatchNorm1d(64)
        self.bn_conv_2 = nn.BatchNorm1d(128)
        self.bn_conv_3 = nn.BatchNorm1d(1024)

        # Fully Connected layers
        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.affine_dim * affine_dim)

        # Batch Norms for FC layers
        self.bn_fc_1 = nn.BatchNorm1d(512)
        self.bn_fc_2 = nn.BatchNorm1d(256)

        # Dropout for FC layers
        self.do_1 = nn.Dropout(p=self.dropout_prop)
        self.do_2 = nn.Dropout(p=self.dropout_prop)


    def forward(self, x):
        """
        Forward propagation of TNetCls

        Args:
        - x: A Tensor of shape (B, N, C)

        Returns: (num_output * num_output) Entries of linear transform matrix
        """

        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        x = x.transpose(2, 1)    # x.shape <- (B, C, N)

        x = F.relu(self.bn_conv_1(self.conv_1(x)))
        x = F.relu(self.bn_conv_2(self.conv_2(x)))
        x = F.relu(self.bn_conv_3(self.conv_3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.do_1(F.relu(self.bn_fc_1(self.fc_1(x))))
        x = self.do_2(F.relu(self.bn_fc_2(self.fc_2(x))))
        x = self.fc_3(x)

        x = x.view(-1, self.affine_dim, self.affine_dim)

        # x.shape = (B, affine_dim, affine_dim)
        return x