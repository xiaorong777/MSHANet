import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from utils.util import Conv2dWithConstraint, LinearWithConstraint
from utils.util import ChannelAttention,SpatialAttention,PEAttention
from torch_geometric.nn import GCNConv
import numpy as np
import torch.nn.functional as F

class GCNConvWithConstraint(nn.Module):
    '''
    A constrained Graph Convolutional Layer inspired by EEGNet for Graph-based models.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        doWeightNorm (bool): If True, applies weight normalization to ensure the weights do not exceed max_norm.
        max_norm (float): The maximum norm for weight normalization.
        **kwargs: Additional arguments for GCNConv.

    Reference:
        Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, in_channels, out_channels, doWeightNorm=True, max_norm=1, **kwargs):
        super(GCNConvWithConstraint, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, **kwargs)
        self.doWeightNorm = doWeightNorm
        self.max_norm = max_norm

    def forward(self, x, edge_index):
        if self.doWeightNorm:
            # Assuming the weight matrix is stored in `self.conv.lin.weight`
            self.conv.lin.weight.data = torch.renorm(self.conv.lin.weight.data, p=2, dim=1, maxnorm=self.max_norm)
        return self.conv(x, edge_index)

class PE_resblock(nn.Module):
    def __init__(self,in_chan=32,in_chan2=64, kerSize_1=(1,8), kerSize_2=(1,16),
                 kerStr=1, out_chan=64,out_chan2=128 ,pool_ker=(3,3), pool_str=1, bias=False, max_norm=1.):
        super(PE_resblock, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kerSize_1,
                stride=kerStr,
                padding='same',
                bias=bias,
                max_norm=max_norm
            ),
            nn.ELU(),
            Conv2dWithConstraint(
                in_channels=out_chan,
                out_channels=out_chan,
                kernel_size=kerSize_2,
                stride=kerStr,
                padding='same',
                bias=bias,
                max_norm=max_norm
            )   
        )
 
        self.av = nn.ELU()

        self.iden1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=max_norm
        )

        self.conv2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=in_chan2,
                out_channels=out_chan2,
                kernel_size=kerSize_1,
                stride=kerStr,
                padding='same',
                bias=bias,
                max_norm=max_norm
            ),
            nn.ELU(),
            Conv2dWithConstraint(
                in_channels=out_chan2,
                out_channels=out_chan2,
                kernel_size=kerSize_2,
                stride=kerStr,
                padding='same',
                bias=bias,
                max_norm=max_norm
            )
        )

        self.iden2 = Conv2dWithConstraint(
            in_channels=in_chan2,
            out_channels=out_chan2,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=max_norm
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.av(x1+self.iden1(x))
        x3 = self.conv2(x2)
        out = x3 +self.iden2(x2)
        out = self.av(out)
        return out
    




    #%% Inception DW Conv layer
class gcconvInception(nn.Module):
    def __init__(self, in_chan=1000, hidden_channels=256,out_channels=64 ,bias=False,pool_ker=(3,3), pool_str=1):
        super(gcconvInception, self).__init__()
        self.conv1 = GCNConvWithConstraint(in_chan, hidden_channels,max_norm=1)  # 第一层卷积
        self.conv2 = LinearWithConstraint(in_features=256,out_features=64,max_norm=.5)  # 第二层卷积
        self.dp =  nn.Dropout(p=0.5)
        self.elu = nn.ELU()

    def create_adjacency_matrix(self,threshold):

        channel_positions = {
        1: (0, 7), 2: (-7, 3.5), 3: (-3.5, 3.5), 4: (0, 3.5), 5: (3.5, 3.5), 6: (7, 3.5),
        7: (-10.5,0), 8: (-7, 0), 9: (-3.5, 0), 10: (0, 0), 11: (3.5, 0), 12: (7, 0), 13: (10.5, 0),
        14: (-7, -3.5), 15: (-3.5,-3.5), 16: (0, -3.5), 17: (3.5, -3.5), 18: (7, -3.5), 19: (-3.5, -7), 20: (0, -7),
        21: (3.5, -7), 22: (0, -10.5)
        }

        coords = np.array([channel_positions[i+1] for i in range(22)])  # 获取位置数组
        n = coords.shape[0]
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
                    if dist < threshold:
                        adj_matrix[i, j] = 1
        # 添加自连接
        np.fill_diagonal(adj_matrix, 1)
        # 将邻接矩阵转换为边索引
        edge_index = np.column_stack(np.where(adj_matrix == 1))  # 获取邻接矩阵中值为 1 的索引
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转换为 PyTorch 张量
        return edge_index

    def forward(self, x):
        edge_index = self.create_adjacency_matrix(threshold=5)
        device = x.device  # 获取 x 的设备
        edge_index = edge_index.to(device)  # 将 edge_index 转移到相同设备
        x, edge_index = x, edge_index
        # 第一层卷积
        x = self.elu(self.conv1(x, edge_index))  # 输出形状: (22, 128)
        x= self.dp(x)
        # 第二层卷积
        x = self.conv2(x)  # 输出形状: (22, 250)
        # out = x.unsqueeze(0)
        return x
    

    #%% Inception DW Conv layer
class convInception(nn.Module):
    def __init__(self, in_chan=1, kerSize_1=(3,3), kerSize_2=(5,5), kerSize_3=(7,7),
                 kerStr=1, out_chan=16, bias=False,pool_ker=(3,3), pool_str=1):
        super(convInception, self).__init__()

        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//4,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )


        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//4,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )

        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//4,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding='same',
            bias=bias,
            max_norm=1.
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(round(pool_ker[0]/2+0.1)-1,round(pool_ker[1]/2+0.1)-1)
        )      
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan//4,
            kernel_size=1,
            stride=1,
            bias=bias,
            max_norm=1.
        )


        # self.point_conv = Conv2dWithConstraint(
        #         in_channels =(out_chan*3)//2,
        #         out_channels=out_chan,
        #         kernel_size =1,
        #         stride      =1,
        #         bias        =False
        #     )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(x)
        out = torch.cat((p1,p2,p3,p4), dim=1)
        return out