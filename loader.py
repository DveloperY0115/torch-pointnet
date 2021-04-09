"""
Custom data loader for PointNet
"""

from torch.utils.data import Dataset, DataLoader

class PointNetDataset(Dataset):

    """
    Dataset object for PointNet
    """

    def __init__(self, num_classes=40):
        pass



def read_off(filename):
    """
    Read .OFF file and convert it to Numpy Array

    Args:
    - filename: String. Name of .OFF file

    Returns: Converted Numpy array
    """
    