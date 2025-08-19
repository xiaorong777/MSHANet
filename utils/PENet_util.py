import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from utils.util import Conv2dWithConstraint, LinearWithConstraint
import numpy as np
import torch.nn.functional as F


    #%% Inception DW Conv layer
class convInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(3,3), kerSize_2=(5,5), kerSize_3=(7,7),
                 kerStr=1, out_chan=16, bias=False,pool_ker=(3,3), pool_str=1):
        super(convInception, self).__init__()

        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//3,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )


        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//3,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )

        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//3,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )


    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        out = torch.cat((p1,p2,p3), dim=1)
        return out
