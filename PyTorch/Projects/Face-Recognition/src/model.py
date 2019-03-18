# import necessary modules
import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable # no more required

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, max_pool_stride=2, \
            dropout_ratio=0.5):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size)
        self.max_pool2d = nn.MaxPool2d(max_pool_stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_ratio)

    def forward(self, X):
        x = self.relu(self.conv2(self.relu(self.conv1(x))))
        return self.dropout(self.max_pool2d(x))

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        self.convs = []
        self.convs.append(ConvLayer(NUM_CHANNELS, 32, kernel_size=5))
        self.convs.append(ConvLayer(32, 64, kernel_size=5))
        conv_output_size, _ = get_convert_output_size( # to be contd
