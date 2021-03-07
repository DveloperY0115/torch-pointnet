import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        for data, label in generate_batch(file):
            print(data.shape, label.shape)



def generate_batch(filename, batch_size=32):
    print('--- Loading From ---')
    print('---{}---'.format(filename))
    data, label = loader.load_datafile(filename)
    data = data[:,0:NUM_POINT,:]
    data, label, _ = loader.shuffle_data(data, np.squeeze(label))
    label = np.squeeze(label)
    file_size = data.shape[0]
    num_batches = file_size // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        # Data augmentation - Random rotation & Gaussian noise
        current_data = loader.rotate_point_cloud(data[start_idx:end_idx, :, :])
        current_data = loader.jitter_point_cloud(current_data)
        current_label = label[start_idx:end_idx]
        yield current_data, current_label


if __name__ == '__main__':

    model = PointNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    train(model, criterion, optimizer)
