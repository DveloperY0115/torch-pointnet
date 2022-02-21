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

try:
    import wandb
except:
    wandb = None

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.pointnet_cls import PointNetCls
from utils.dataset import PointNetDataset
from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

parser = argparse.ArgumentParser(description="Parsing argument")
parser.add_argument("--device_id", type=int, default=0, help="ID of GPU to be used")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
parser.add_argument(
    "--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
)
parser.add_argument("--step_size", type=int, default=100, help="Step size of StepLR")
parser.add_argument("--gamma", type=float, default=0.99, help="Gamma of StepLR")
parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument(
    "--num_iter", type=int, default=100, help="Number of iteration in one epoch"
)
parser.add_argument(
    "--batch_size", type=int, default=450, help="Size of a batch per device"
)
parser.add_argument(
    "--num_worker",
    type=int,
    default=2,
    help="Number of workers for data loader per device",
)
parser.add_argument(
    "--local_rank", type=int, default=0, help="Local rank for distributed training"
)
parser.add_argument(
    "--out_dir", type=str, default="out", help="Name of the output directory"
)
parser.add_argument(
    "--save_period", type=int, default=50, help="Number of epochs between checkpoints"
)
parser.add_argument(
    "--vis_period",
    type=int,
    default=10,
    help="Number of epochs between each visualization",
)
args = parser.parse_args()


def main():
    device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    # check GPU
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=n_gpu,
            init_method="env://",
        )
        synchronize()

    # initialize W&B after initializing torch.distributed
    if get_rank() == 0 and wandb:
        wandb.init(project="torch-pointnet-distributed", config=args)

    # model & optimizer, schedulers
    network = PointNetCls().to(device)

    optimizer = optim.Adam(
        network.parameters(),
        betas=(args.beta1, args.beta2),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    if args.distributed:
        network = DDP(
            network,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # prepare data loaders
    train_dataset = PointNetDataset(mode="train")
    test_dataset = PointNetDataset(mode="test")

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=n_gpu,
            rank=args.local_rank,
        )
        if args.distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=n_gpu,
            rank=args.local_rank,
        )
        if args.distributed
        else None
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_worker,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=(test_sampler is None),
        num_workers=args.num_worker,
        sampler=test_sampler,
    )

    pbar = range(args.num_epoch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    # run training
    for epoch in pbar:

        if epoch > args.num_epoch:
            print("[!] Training finished!")
            break

        if args.distributed:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            network, optimizer, scheduler, device, train_loader, epoch
        )

        with torch.no_grad():
            test_loss, test_accuracy, fig = run_test(
                network, device, test_loader, epoch
            )

        # log data
        if get_rank() == 0:
            print("------------------------------")
            print("Epoch {} training avg_loss: {}".format(epoch, avg_loss))
            print("Epoch {} test loss: {}".format(epoch, test_loss))
            print("Epoch {} accuracy: {} %".format(epoch, test_accuracy))
            print("------------------------------")

            if wandb:
                logged_data = {
                    "Train/Loss": avg_loss,
                    "Test/Loss": test_loss,
                    "Test/Accuracy": test_accuracy,
                }

                if epoch != 0 and ((epoch + 1) % args.vis_period == 0):
                    logged_data["Test/Visualization"] = wandb.Image(fig)

                wandb.log(logged_data, step=epoch + 1)

        if epoch != 0 and ((epoch + 1) % args.save_period == 0):
            # TODO: Synchronization across devices after saving & loading
            save_dir = args.out_dir

            if get_rank() == 0:
                # save model on only one process
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
                    os.path.join(save_dir, "{}.pt".format(str(epoch))),
                )
                print(
                    "[!] Saved model at: {}".format(
                        os.path.join(save_dir, "{}.pt".format(str(epoch)))
                    )
                )

            # wait until the checkpoint is saved to disk
            synchronize()
            map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}

            # load the same parameters on all processes
            ckpt_dict = torch.load(
                os.path.join(save_dir, "{}.pt".format(str(epoch))),
                map_location=map_location,
            )
            epoch = ckpt_dict["epoch"]
            network.load_state_dict(ckpt_dict["model_state_dict"])
            optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])

    # clean up
    save_dir = args.out_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
        },
        os.path.join(save_dir, "final.pt"),
    )

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
    total_loss = torch.zeros(1, device=device)

    for batch in train_loader:
        data, label = batch

        # initialize gradient
        optimizer.zero_grad()

        # send data to device
        data = data.to(device)
        label = label.to(device).long().squeeze()

        # forward propagation
        pred = network(data)

        # calculate loss
        loss = nn.CrossEntropyLoss()(pred, label)

        # back propagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss

    # collect loss all over the devices
    with torch.no_grad():
        total_loss_reduced = reduce_sum(total_loss)

        avg_loss = total_loss_reduced.mean().item() / len(train_loader)

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
    loss = reduce_sum(loss).item()

    # calculate accuracy
    pred = torch.argmax(pred, dim=1)
    answer = (pred == label).type(torch.uint8)
    num_correct = torch.sum(answer)
    accuracy = (num_correct / args.batch_size) * 100
    accuracy = reduce_sum(accuracy) / get_world_size()

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
    fig, ax = plt.subplots(
        nrows=4,
        ncols=4,
        figsize=(20, 20),
        subplot_kw=dict(projection="3d"),
        constrained_layout=True,
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

    # close figure before return
    plt.close(fig)

    return image


if __name__ == "__main__":
    main()
