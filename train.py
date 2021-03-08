from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
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
MODELNET_CLASSES = 40

epoch = 0

def save_checkpoint(path, model, criterion, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion
    }, path)


def train(model,
          criterion,
          optimizer,
          num_epochs=300,
          checkpoint_dir=str(os.path.join(BASE_DIR, 'checkpoints'))):

    global epoch

    filename = os.path.join(checkpoint_dir, 'checkpoint.tar')

    for epoch in range(num_epochs):
        save_checkpoint(filename, model, criterion, optimizer, epoch)
        train_one_epoch(model, criterion, optimizer)
        save_checkpoint(filename, model, criterion, optimizer, epoch+1)


def train_one_epoch(model, criterion, optimizer):
    global epoch
    files = TRAIN_FILES
    random.shuffle(files)
    cost_sum = 0
    num_data = 0

    for (idx, file) in enumerate(files):
        print('Iterating over dataset: [{}/{}]'.format(idx, len(files)))
        data, label = generate_dataset(file)
        dataset = TensorDataset(data, label)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
        num_data += len(loader)
        for x, y in tqdm(loader):
            # Make prediction
            pred = model(x)

            # Calculate loss
            cost = criterion(pred, y)

            # Update weights
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            cost_sum += cost.item()

    print('------------epoch {}----------------'.format(epoch))
    print('Cost: {}'.format(cost_sum / num_data))
    for param_group in optimizer.param_groups:
        print('Learning rate: {}'.format(str(param_group['lr'])))
    print('----------------*-------------------')


def generate_dataset(filename):
    data, label = loader.load_datafile(filename)
    data = data[:,0:NUM_POINT,:]
    label = np.squeeze(label)

    # data augmentation
    data = loader.rotate_point_cloud(data[:, :, :])
    data = loader.jitter_point_cloud(data)

    return torch.from_numpy(data), torch.LongTensor(label)


if __name__ == '__main__':

    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set directory for storing weight files
    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.tar')

    model = PointNet(num_classes=MODELNET_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                     lr_lambda=lambda epoch: 0.95**epoch)

    if os.path.exists(checkpoint_file):
        print('Weight file already exists. Loading...')

        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        print('On epoch: {}'.format(epoch))
        model.train()
    else:
        os.mkdir('./checkpoints')

    summary(model, (1024, 3))

    train(model, criterion, optimizer)
