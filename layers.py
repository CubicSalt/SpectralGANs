import math

import torch.nn as nn

import numpy as np

class Identity(nn.Module):
    def __init__(self, *args, **keyword_args):
        super().__init__()

    def forward(self, x):
        return x

class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
    
class WScaleLayer(nn.Module):
    def __init__(self, size, fan_in, gain=np.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / np.sqrt(fan_in) # No longer a parameter
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])
        return x

def _get_norm_layer_2d(norm, num_features):
    if norm == 'none':
        return Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(num_features)
    elif norm == 'layer_norm':
        return nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError