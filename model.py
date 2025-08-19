import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary
from torchstat import stat
from utils.util import Conv2dWithConstraint, LinearWithConstraint,DSEAttention
from utils.util import ChannelAttention,SpatialAttention,DCTAttention
from utils.TCN_util import TemporalConvNet
from utils.MHSA_util import TalkingHeadSelfAttention
from utils.PENet_util import convInception
from dataLoad.preprocess import create_adjacency_matrix


#%%

class model(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, dropoutRate=0.5, kerSize_Tem=16,kerSize=32,F1=24, D=2,
                 tcn_filters=64, tcn_kernelSize=4, tcn_dropout=0.3, bias=False, n_classes=4):
        super(model, self).__init__()
        F2 = F1*D
        self.channel_positions = {
        1: (0, 7), 2: (-7, 3.5), 3: (-3.5, 3.5), 4: (0, 3.5), 5: (3.5, 3.5), 6: (7, 3.5),
        7: (-10.5,0), 8: (-7, 0), 9: (-3.5, 0), 10: (0, 0), 11: (3.5, 0), 12: (7, 0), 13: (10.5, 0),
        14: (-7, -3.5), 15: (-3.5,-3.5), 16: (0, -3.5), 17: (3.5, -3.5), 18: (7, -3.5), 19: (-3.5, -7), 20: (0, -7),
        21: (3.5, -7), 22: (0, -10.5)
        }
        self.channel_positions_2b = {
        1: (-3.5, 0), 2: (0, 0), 3: (3.5, 0)
        }


        self.block1 = nn.Sequential(
            convInception(
                in_chan=1,
                kerSize_1=(1,16),
                kerSize_2=(1,32),
                kerSize_3=(1,64),
                out_chan=F1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=F1),
        )

        self.depthwiseConv = nn.Sequential( 
            Conv2dWithConstraint(
                in_channels=F1, 
                out_channels=F1*D,
                kernel_size=(eeg_chans, 1),
                groups=F1,
                bias=bias, 
                max_norm=1.
            ), 
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.seqarableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F2, 
                out_channels=F2,
                kernel_size=(1,kerSize_Tem),
                stride=1,
                padding='same',
                groups=F2,
                bias=bias
            ),
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1,1),
                stride=1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.flatten_eeg = nn.Flatten()


        self.layerNorm = nn.LayerNorm(
            normalized_shape=F2,
            eps=1e-6
        )
        self.multihead_attn = TalkingHeadSelfAttention(
            embed_dim = F2,
            heads     = 6,
            dropout   = 0.3,
            norm      = .25
        )

        self.dse = DSEAttention(F2)

        self.dctfa = DCTAttation()

        self.tcn_block = TemporalConvNet(
            num_inputs  = F2*2,
            num_channels= [tcn_filters, tcn_filters],
            kernel_size = tcn_kernelSize,
            dropout     = tcn_dropout,
            bias        = False,
            WeightNorm  = True,
            max_norm    = .5
        )

        self.fal_tcn = nn.Flatten()
        self.fal_am = nn.Flatten()

        self.class_head = nn.Sequential(
            LinearWithConstraint(
                in_features=784,
                out_features=n_classes,
                max_norm=.25,
                bias = True
            )
        )
        self.adj_matrix = nn.Parameter(torch.randn(eeg_chans, eeg_chans))
        self.softmax = nn.Softmax(dim=-1)

    def gaussian_adjacency_matrix(self,data,channel_positions,sigma=2.0,device='cuda:0'):
        coords = np.array([channel_positions[i+1] for i in range(22)])  # 获取位置数组
        num_channels = coords.shape[0]
        distance_matrix = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                distance_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        gaussian_encoding_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
        pos =  torch.tensor(gaussian_encoding_matrix).float().to(device)
        return pos
    
    def gaussian_adjacency_matrix_2b(self,data,channel_positions,sigma=1.0,device='cuda:0'):
        coords = np.array([channel_positions[i+1] for i in range(3)])  # 获取位置数组
        num_channels = coords.shape[0]
        distance_matrix = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            for j in range(num_channels):
                distance_matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        gaussian_encoding_matrix = np.exp(-distance_matrix**2 / (2 * sigma**2))
        pos =  torch.tensor(gaussian_encoding_matrix).float().to(device)
        return pos




    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        pos = self.gaussian_adjacency_matrix(x,channel_positions=self.channel_positions)
        adj_matrix = (self.adj_matrix + self.adj_matrix.T) / 2  # 对称化
        adj_matrix = torch.sigmoid(pos)
        x_am = torch.matmul(adj_matrix, x)
        x1 = self.block1(x)
        x1 = self.dctfa(x1)
        x2 = self.depthwiseConv(x1)
        x2 = self.seqarableConv(x2)


        p1 = torch.squeeze(x2, dim=2) # (batch, F1*D, 15)
        p2 = torch.transpose(p1, len(p1.shape)-2, len(p1.shape)-1) # (batch, 15, F1*D)
        p2 = self.layerNorm(p2)
        p2, attention_scores = self.talkinghead_attn(p2)
        p2 = torch.transpose(p2, len(p2.shape)-2, len(p2.shape)-1) # (batch, F1*D, 15)
        p2 = torch.squeeze(p2, dim=2) # (batch, F1*D, 15)


        x4 = self.dse(x2)
        x4 = torch.squeeze(x4, dim=2) # NCW
        x4 = self.fal_am(x4)
        p3 = torch.cat((p1,p2),dim=1)

        p3 = self.tcn_block(p3)
        p3 = p3[:, :, -1] # NC
        out_tcn = self.fal_tcn(p3)
        out_tcn = torch.cat((x4, out_tcn), dim=-1)
        # # 注册钩子
        # hook_handle = out_tcn.register_hook(get_activation('out_tcn'))
        out_tcn = self.class_head(out_tcn)
        out = self.softmax(out_tcn)
        return out


#%%
###============================ Initialization parameters ============================###
channels = 22
samples  = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = model()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))

if __name__ == "__main__":
    main()

