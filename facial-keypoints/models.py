from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    @staticmethod
    def convolution_block(
        channels_in, 
        channels_out, 
        dropout, 
        suffix,
        conv_kernel_size=4,
        pooling_kernel_size=2):
        
        return nn.Sequential(OrderedDict([
            (f"conv_{suffix}", nn.Conv2d(channels_in, channels_out, conv_kernel_size)),
            (f"batch_norm_{suffix}", nn.BatchNorm2d(channels_out)),
            (f"activ_{suffix}", nn.ReLU()),
#             (f"conv2_{suffix}", nn.Conv2d(channels_out, channels_out, conv_kernel_size, padding=(conv_kernel_size - 1) // 2)),
#             (f"activ2_{suffix}", nn.ELU()),
            (f"pool_{suffix}", nn.MaxPool2d(pooling_kernel_size)),
#             (f"dropout_{suffix}", nn.Dropout(dropout))
        ]))
        
    @staticmethod
    def dense_block(in_features, out_features, dropout, suffix):
        return nn.Sequential(OrderedDict([
            (f"dense_{suffix}", nn.Linear(in_features, out_features)),
            (f"batch_norm_{suffix}", nn.BatchNorm1d(out_features)),
            (f"activ_{suffix}", nn.ReLU()),
#             (f"dropout_{suffix}", nn.Dropout(dropout))
        ]))
        
    def __init__(self):
        super(Net, self).__init__()
        
        self.convolution_part = nn.Sequential(OrderedDict([
            ("conv_block_0", Net.convolution_block(3, 16, 0.1, "conv1", conv_kernel_size=5)),
            ("conv_block_1", Net.convolution_block(16, 32, 0.2, "conv1", conv_kernel_size=4)),
            ("conv_block_2", Net.convolution_block(32, 64, 0.3, "conv2", conv_kernel_size=3)),
            ("conv_block_3", Net.convolution_block(64, 128, 0.4, "conv3", conv_kernel_size=3)),
            ("conv_block_4", Net.convolution_block(128, 256, 0.5, "conv4", conv_kernel_size=1))
        ]))
        
        self.flatten = Flatten()
        
        self.dense_part = nn.Sequential(OrderedDict([
            ("dense_block_1", Net.dense_block(6400, 1000, 0.6, "dense1")),
            ("dense_block_2", Net.dense_block(1000, 1000, 0.7, "dense2"))
        ]))
        
        self.classification_part = nn.Linear(1000, 136)
        
        self.whole_network = nn.Sequential(OrderedDict([
            ("convolution_part", self.convolution_part),
            ("flatten", self.flatten),
            ("dense_part", self.dense_part),
            ("classification_part", self.classification_part)
        ]))
        
    def forward(self, x):
        return self.whole_network(x)