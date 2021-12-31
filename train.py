"""
tran.py

Training routine for PointNet.
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.pointnet_cls import PointNetCls
from utils.dataset import PointNetDataset

parser = argparse.ArgumentParser(description="Parsing argument")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
parser.add_argument("--step_size", type=int, default=100, help="Step size of StepLR")
parser.add_argument("--gamma", type=float, default=0.99, help="Gamma of StepLR")
parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
parser.add_argument("--num_worker", type=int, default=8, help="Number of workers for data loader")
parser.add_argument("--out_dir", type=str, default="out", help="Name of the output directory")
parser.add_argument(
    "--save_period", type=int, default=50, help="Number of epochs between checkpoints"
)
parser.add_argument(
    "--vis_period", type=int, default=1, help="Number of epochs between each visualization"
)
args = parser.parse_args()


def main():
    
    config = wandb.config

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model & optimizer, schedulers
    network = PointNetCls().to(device)

    # make W&B track model's gradient and topology
    wandb.watch(network, log_freq=100, log_graph=False)

    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("[!] Multiple GPU available, but not yet supported")

    optimizer = optim.Adam(network.parameters(), betas=(config["beta1"], config["beta2"]), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # prepare data loaders
    train_data = PointNetDataset(mode="train")
    test_data = PointNetDataset(mode="test")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_worker"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # run training
    for epoch in tqdm(range(config["num_epoch"]), leave=False):
        avg_loss = train_one_epoch(network, optimizer, scheduler, device, train_loader, epoch)

        # log data
        wandb.log({"Train/Loss": avg_loss}, step=epoch)

        print("------------------------------")
        print("Epoch {} training avg_loss: {}".format(epoch, avg_loss))
        print("------------------------------")

        with torch.no_grad():
            test_loss, test_accuracy, fig = run_test(network, device, test_loader, epoch)

            logged_data = {"Test/Loss": test_loss, "Test/Accuracy": test_accuracy}

            if epoch != 0 and ((epoch + 1) % args.vis_period == 0):
                logged_data["Test/Visualization"] = wandb.Image(fig)

            wandb.log(
                logged_data, step=epoch
            )

            print("------------------------------")
            print("Epoch {} test loss: {}".format(epoch, test_loss))
            print("Epoch {} accuracy: {} %".format(epoch, test_accuracy))
            print("------------------------------")

        if epoch != 0 and ((epoch + 1) % args.save_period == 0):
            # save model
            save_dir = args.out_dir

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                },
                os.path.join(save_dir, "{}.pt".format(str(epoch + 1))),
            )

            print(
                "[!] Saved model at: {}".format(os.path.join(save_dir, "{}.pt".format(str(epoch))))
            )

    # clean up
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save({"model_state_dict": network.state_dict(),}, os.path.join(save_dir, "final.pt"))

    print("[!] Saved model at: {}".format(os.path.join(save_dir, "final.pt")))


def train_one_epoch(network, optimizer, scheduler, device, train_loader, epoch):
    """
    Training loop for an epoch.

    Args:
    - network (torch.nn.Module): Pytorch model being optimized.
    - optimizer (torch.optim.Optimizer): Optimizer used for training.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - device (torch.device): Object representing currently used device.
    - train_loader (torch.utils.data.DataLoader): Dataloader for training data.
    - epoch (int): Index of current epoch.

    Returns:
    - avg_loss (float): Average loss computed over an epoch.
    """
    train_iter = iter(train_loader)

    total_loss = 0

    for _ in tqdm(range(args.num_iter), leave=False):
        try:
            data, label = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, label = next(train_iter)

        # initialize gradient
        optimizer.zero_grad()

        # send data to device
        data = data.to(device)
        label = label.to(device).long().squeeze()

        # forward propagation
        pred = network(data)

        # calculate loss
        loss = nn.CrossEntropyLoss()(pred, label)
        total_loss += loss.item()

        # back propagation
        loss.backward()
        optimizer.step()

        # adjust learning rate
        scheduler.step()

    avg_loss = total_loss / args.num_iter

    return avg_loss


def run_test(network, device, test_loader, epoch):
    """
    Testing loop for an epoch.

    Args:
    - network (torch.nn.Module): Pytorch model being optimized.
    - device (torch.device): Object representing currently used device.
    - test_loader (torch.utils.data.DataLoader): Dataloader for test data.
    - epoch (int): Index of current epoch.

    Returns:
    - avg_loss (float): Test loss.
    - accuracy (float): Classification accuracy.
    """
    test_iter = iter(test_loader)

    data, label = next(test_iter)

    # send data to device
    data = data.to(device)
    label = label.to(device).long().squeeze()

    # forward propagation
    pred = network(data)

    # calculate loss
    loss = nn.CrossEntropyLoss()(pred, label)

    # calculate accuracy
    pred = torch.argmax(pred, dim=1)
    answer = (pred == label).type(torch.uint8)
    num_correct = torch.sum(answer)
    accuracy = (num_correct / args.batch_size) * 100

    # plot point cloud and their predicted / GT labels
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()

    fig = plot_pc_labels(data, (pred, label))

    return loss, accuracy, fig


def plot_pc_labels(pc, labels):
    """
    Plot point cloud and their corresponding labels.

    Args:
    - pc (torch.Tensor): Tensor of shape (B, N, 3). Batch of point clouds.
    - labels (Tuple): Tuple containing tensors representing labels of point clouds, such as predicted / GT class names.

    Returns:
    - image (np.array): A Numpy array representing the image of rendered figure
    """
    # get class ID
    pred_id = labels[0]
    gt_id = labels[1]

    # create ID -> class name table
    cls_names = {}
    with open("data/modelnet40_ply_hdf5_2048/shape_names.txt", "r") as f:
        lines = f.readlines()

        for idx, line in enumerate(lines):
            cls_names[idx] = line.strip()

    # plot point clouds and labels
    fig, ax = plt.subplots(nrows=4, ncols=4, 
        figsize=(20, 20), subplot_kw=dict(projection="3d"),
        constrained_layout=True
        )

    # get the canvas corresponding to the figure being drawn
    canvas = FigureCanvas(fig)
    width, height = fig.get_size_inches() * fig.get_dpi()

    for idx, axi in enumerate(ax.flat):
        if idx < len(pc):
            axi.scatter(pc[idx, :, 0], pc[idx, :, 1], pc[idx, :, 2])
            axi.set_title(
                "Pred: {} | GT: {}".format(
                    cls_names[pred_id[idx].item()], cls_names[gt_id[idx].item()]
                )
            )

    # draw the canvas
    canvas.draw()

    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = np.reshape(image, (int(height), int(width), 3))
    return image


if __name__ == "__main__":
    wandb.init(project="torch-pointnet", config=args)
    main()
