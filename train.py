import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from models.pointnet_cls import PointNet
import loader
import random
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

TRAIN_FILES = loader.get_datafiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = loader.get_datafiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024


def train(model, criterion, optimizer, num_epochs=300):
    for i in range(num_epochs):
        train_one_epoch(model, criterion, optimizer)
        pass


def train_one_epoch(model, criterion, optimizer):
    files = TRAIN_FILES
    random.shuffle(files)

    for file in files:
        data, label = generate_dataset(file)
        dataset = TensorDataset(data, label)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
        for x, y in loader:
            # Make prediction
            pred = model(x)

            # Calculate loss
            cost = criterion(pred, y)

            # Update weights
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print('Cost: {}'.format(cost.item()))


def generate_dataset(filename):
    print('--- Loading From ---')
    print('---{}---'.format(filename))
    data, label = loader.load_datafile(filename)
    data = data[:,0:NUM_POINT,:]
    label = np.squeeze(label)

    # data augmentation
    data = loader.rotate_point_cloud(data[:, :, :])
    data = loader.jitter_point_cloud(data)

    print(data.shape, label.shape)
    return torch.from_numpy(data), torch.from_numpy(label).view(-1, 1)


if __name__ == '__main__':

    model = PointNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train(model, criterion, optimizer)
