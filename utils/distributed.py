from typing import Iterable
import pickle

import torch
import torch.nn as nn
from torch import distributed as dist

def get_rank(group=None) -> int:
    """
    Wrapper for torch.distributed.get_rank().

    Checks whether
    (1) torch.distributed is available on the calling machine
    (2) torch.distributed is initialized before invoking this function.

    Args:
    - group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
    - The rank of the process group
        -1, if not part of the group
    """
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank(group)

def synchronize(group=None) -> None:
    """
    Synchronize processes in the given process group.

    Checks whether
    (1) torch.distributed is available on the calling machine
    (2) torch.distributed is initialized before invoking this function.
    
    Args:
    - group (ProcessGroup, optional): The process group to work on. 
        If None, the default process group will be used.

    Returns:
    - Async work handle, if async_op is set to True. 
        None, if not async_op or if not part of the group
    """
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = get_world_size(group)

    if world_size == 1:
        # single process is running in the group,
        # no need to synchronize
        return
    
    # otherwise, block until other processes finish
    # their job and synchronize
    dist.barrier()

def get_world_size(group=None) -> int:
    """
    Get the number of processes in the given process group.

    Checks whether
    (1) torch.distributed is available on the calling machine
    (2) torch.distributed is initialized before invoking this function.
    
    Args:
    - group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
    - The world size of the process group 
        -1, if not part of the group
    """
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size(group)

def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce the tensor data from all machines to get
    the final tensor.

    Args:
    - tensor (torch.Tensor): A Pytorch tensor.

    Returns:
    - A tensor reduced across all machines.
    """
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor
    
    # reduce the tensor across all machines
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

# TODO: Investigate those functions!
def gather_grad(params: Iterable[nn.Parameter]) -> None:
    """
    Collect gradient across all machines and compute their average.

    Args:
    - params (Iterable of torch.nn.Parameters):
    """
    world_size = get_world_size()

    if world_size == 1:
        return
    
    for param in params:
        if param.grad is not None:
            # reduce gradients of parameters across machines
            # and eventually compute the average gradient
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)  # in-place division

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        # only a single device is being used
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses
