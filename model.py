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
from utils.util import DCTFCAttention
from utils.TCN_util import TemporalConvNet
from utils.MHSA_util import TalkingHeadSelfAttention
from utils.PENet_util import TemconvInception

#%%
class EEGSpatialEncoder(nn.Module):
    def __init__(self, channel_positions: dict, distance_metric='euclidean'):
        super(EEGSpatialEncoder, self).__init__()
        self.num_channels = len(channel_positions)
        self.last_attn_weights = None

        positions = torch.tensor([channel_positions[i+1] for i in range(self.num_channels)], dtype=torch.float32)
        self.register_buffer('positions', positions)
        self.register_buffer('adj_matrix', self._compute_adjacency(positions, metric=distance_metric))
        self.spatial_proj = nn.Linear(self.num_channels, self.num_channels, bias=False)

    def _compute_adjacency(self, pos, metric='euclidean'):
        if metric == 'euclidean':
            dists = torch.cdist(pos, pos, p=2)  # [C, C]
        elif metric == 'cosine':
            norm = F.normalize(pos, dim=-1)
            dists = 1 - torch.matmul(norm, norm.T)
        else:
            raise ValueError("Unsupported metric")

        sigma = torch.std(dists)
        sim = torch.exp(-dists**2 / (2 * sigma**2))  # shape [C, C]
        return sim

    def forward(self, x):
        """
        x: Tensor shape [B, C=22, T]
        return: same shape [B, 22, T]
        """
        if len(x.shape) != 3:
            x = torch.squeeze(x, 1)
        B, C, T = x.shape
        weight = self.spatial_proj(self.adj_matrix)  # [C, C]
        x_out = torch.matmul(weight, x)  # [B, C, T]
        x_out = x_out.unsqueeze(1)
        return x_out


class model(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, dropoutRate=0.5, kerSize_Tem=16,kerSize=32,F1=24, D=2,
                 tcn_filters=32, tcn_kernelSize=4, tcn_dropout=0.3, bias=False, n_classes=4):
        super(model, self).__init__()
        F2 = F1*D
        self.channel_positions_2a = {
        1: (0, 7), 2: (-7, 3.5), 3: (-3.5, 3.5), 4: (0, 3.5), 5: (3.5, 3.5), 6: (7, 3.5),
        7: (-10.5,0), 8: (-7, 0), 9: (-3.5, 0), 10: (0, 0), 11: (3.5, 0), 12: (7, 0), 13: (10.5, 0),
        14: (-7, -3.5), 15: (-3.5,-3.5), 16: (0, -3.5), 17: (3.5, -3.5), 18: (7, -3.5), 19: (-3.5, -7), 20: (0, -7),
        21: (3.5, -7), 22: (0, -10.5)
        }
        self.channel_positions_2b = {
        1: (-3.5, 0), 2: (0, 0), 3: (3.5, 0)
        }

        self.eegSpatialEncoder = EEGSpatialEncoder(channel_positions=self.channel_positions_2a, distance_metric='euclidean')
        self.block1 = nn.Sequential(
            TemconvInception(
                in_chan=1,
                kerSize_1=(1,16),
                kerSize_2=(1,32),
                kerSize_3=(1,64),
                out_chan=F1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=F1),
        )
     self.DCTFCA = DCTFCAttention(F1)
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
        self.THSA = TalkingHeadSelfAttention(
            embed_dim = F2,
            num_heads  = 6,
            dropout   = 0.3
        )

        self.DSEA = DSEAttention(F2)

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

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        if len(x.shape) is not 4:
            x = torch.unsqueeze(x, 1)
        x = self.eegSpatialEncoder(x)
        x1 = self.block1(x)
        x1 = self.DCTFCA(x1)
        x2 = self.depthwiseConv(x1)
        x2 = self.seqarableConv(x2)


        p1 = torch.squeeze(x2, dim=2) # (batch, F1*D, 15)
        p2 = torch.transpose(p1, len(p1.shape)-2, len(p1.shape)-1) # (batch, 15, F1*D)
        p2 = self.layerNorm(p2)
        p2,attn_logits,attn_scores,attn_weights= self.THSA(p2)
        self.attn_logits = attn_logits.detach()  
        self.attn_scores = attn_scores.detach()  
        self.attn_weights = attn_weights.detach()  
        self.pre_softmax_mix = self.multihead_attn.pre_softmax_mix.detach()  
        self.post_softmax_mix = self.multihead_attn.post_softmax_mix.detach()
        self.pre_softmax_mix
        p2 = torch.transpose(p2, len(p2.shape)-2, len(p2.shape)-1) # (batch, F1*D, 15)
        p2 = torch.squeeze(p2, dim=2) # (batch, F1*D, 15)
        p2 = self.pool(p2)


        x4 = self.DSEA(x2)
        x4 = torch.squeeze(x4, dim=2) # NCW
        x4 = self.fal_am(x4)
        p3 = torch.cat((p1,p2),dim=1)

        p3 = self.tcn_block(p3)
        p3 = p3[:, :, -1] # NC
        out_tcn = self.fal_tcn(p3)
        out = torch.cat((x4, out_tcn), dim=-1)
        out = self.class_head(out)
        out = self.softmax(out)
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







