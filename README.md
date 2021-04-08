# pointnet-torch
Pytorch implementation of PointNet (*Charles R. Q et al., CVPR 2017*)

**NOTE**

This implementation follows the one introduced in the authors' paper, thus not optimized for practical use.  
For instance, 
- most of MLPs in the network are actually replaced to 1D convolution layers.
- T-Nets are not fully implemented yet (will be soon)
- Batch normalization and other methods for preventing gradient vanishing problems are not applied.