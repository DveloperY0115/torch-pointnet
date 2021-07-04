"""
Custom data loader for PointNet
"""

import os
import torch
from torch.utils.data import Dataset

from utils.data_prep_util import load_h5


class PointNetDataset(Dataset):
    def __init__(self, basepath="./data/modelnet40_ply_hdf5_2048", mode="train"):
        """
        Initialize dataset for PointNet.

        Args:
        - basepath (str): Path to where data files are located.
        - mode (str): Selector for which dataset to be loaded. Either 'train' or 'test'.
        """

        self.basepath = basepath
        self.mode = mode

        filenames = []

        if mode == "train":
            f = open(os.path.join(basepath, "train_files.txt"), "r")
        else:
            f = open(os.path.join(basepath, "test_files.txt"), "r")

        # identify files
        files = f.readlines()
        for file in files:
            filenames.append(file.rstrip("\n"))

        self.data = None
        self.labels = None

        # read files and load contents
        for file in filenames:
            data_tmp, label_tmp = load_h5(file)
            if self.data is None:
                self.data = torch.Tensor(data_tmp)
                self.labels = torch.Tensor(label_tmp)
            else:
                self.data = torch.cat((self.data, torch.Tensor(data_tmp)), dim=0)
                self.labels = torch.cat((self.labels, torch.Tensor(label_tmp)), dim=0)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size()[0]
