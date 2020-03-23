from torch import nn
import numpy as np
from layers import *


class PixelCNN(nn.Module):
    def __init__(self, n_channels=32, n_layers=7):
        super(PixelCNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(CondGatedMaskedConv2d('A', 1, n_channels,
                                                 7, 1, 3, bias=False))
        self.layers.append(nn.BatchNorm2d(n_channels))
        self.layers.append(nn.ReLU())
        
        for i in range(1, n_layers+1):
            self.layers.append(CondGatedMaskedConv2d('B', n_channels,
                                                     n_channels, 7, 1, 3,
                                                     bias=False))
            self.layers.append(nn.BatchNorm2d(n_channels))
            self.layers.append(nn.ReLU())
    
        self.layers.append(nn.Conv2d(n_channels, 256, 1))
    
    def forward(self, x, h):
        out = x
        for layer in self.layers:
            if isinstance(layer, CondGatedMaskedConv2d):
                out = layer(out, h)
            else:
                out = layer(out)
        return out
                              

class LabelNet(nn.Module):
    def __init__(self, input_shape=10, output_shape=(28,28)):
        super(LabelNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.linear = nn.Linear(10, np.prod(output_shape))

    def forward(self, h):
        return self.linear(h).view(-1, 1, *self.output_shape)
