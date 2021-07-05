"""
tran.py

Training routine for PointNet.
"""

import os
import argparse
from tqdm import tqdm
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
parser.add_argument("--num_epoch", type=int, default=1000, help="Number of epochs")
parser.add_argument("--num_iter", type=int, default=100, help="Number of iteration in one epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Size of a batch")
parser.add_argument(
    "--save_period", type=int, default=100, help="Number of epochs between checkpoints"
)
args = parser.parse_args()


def main():

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model & optimizer, schedulers
    network = PointNetCls().to(device)

    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("[!] Multiple GPU available, but not yet supported")

    optimizer = optim.Adam(network.parameters(), betas=(args.beta1, args.beta2), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # prepare data loaders
    train_data = PointNetDataset(mode="train")
    test_data = PointNetDataset(mode="test")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # run training
    for epoch in tqdm(range(args.num_epoch)):
        avg_loss = train_one_epoch(network, optimizer, scheduler, device, train_loader)

        print("------------------------------")
        print("Epoch {} training avg_loss: {}".format(epoch, avg_loss))
        print("------------------------------")

        with torch.no_grad():
            test_loss, test_accuracy = run_test(network, device, test_loader)
            print("------------------------------")
            print("Epoch {} test loss: {}".format(epoch, test_loss))
            print("Epoch {} accuracy: {} %".format(epoch, test_accuracy))
            print("------------------------------")

        if epoch != 0 and (epoch % args.save_period == 0):
            # save model
            save_dir = "./checkpoint"

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
                "[!] Saved model at: {}".format(
                    os.path.join(save_dir, "{}.pt".format(str(epoch + 1)))
                )
            )

    # clean up
    save_dir = "./checkpoint"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save({"model_state_dict": network.state_dict(),}, os.path.join(save_dir, "final.pt"))

    print("[!] Saved model at: {}".format(os.path.join(save_dir, "final.pt")))


def train_one_epoch(network, optimizer, scheduler, device, train_loader):

    train_iter = iter(train_loader)

    total_loss = 0
    for _ in tqdm(range(args.num_iter)):
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

    return total_loss / args.num_iter  # average loss


def run_test(network, device, test_loader):

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

    return loss, accuracy


if __name__ == "__main__":
    main()
