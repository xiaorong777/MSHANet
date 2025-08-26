import torch
import torch.nn as nn
from torch.autograd import Function



#%%
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#%%
class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)
            
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

#%%
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

#%%


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    


class PEAttention(nn.Module):
    def __init__(self, kernel_size=(3,3)):
        super(PEAttention, self).__init__()
        
        self.maxpol1 = nn.MaxPool2d(kernel_size=(3,3),stride=1,padding=(1,1))
        self.maxpol2 = nn.MaxPool2d(kernel_size=(5,5),stride=1,padding=(2,2))
        self.maxpol3 = nn.MaxPool2d(kernel_size=(7,7),stride=1,padding=(3,3))
        self.conv1 = nn.Conv2d(3, 1, kernel_size=(1,1),stride=1,padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        max_out1 = self.maxpol1(avg_out)
        max_out2 = self.maxpol2(avg_out)
        max_out3 = self.maxpol3(avg_out)
        out = torch.cat([max_out1, max_out2,max_out3], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)
    

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  #  (B, C, 1, W) 
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  #  (batch_size, channels, 1, width)
        y = self.avg_pool(x).view(b, c)  #  (batch_size, channels)
        y = self.fc(y).view(b, c, 1, 1)  #  (batch_size, channels, 1, 1)
        return x * y.expand_as(x)  # 

class DSEAttention(nn.Module):
    def __init__(self, channels, reduction_list=[4, 8, 16]):
        super(DSEAttention, self).__init__()
        self.branches = nn.ModuleList([
           SEAttention(channels, r) for r in reduction_list
        ])
        self.num_branches = len(reduction_list)

        self.gate = nn.Sequential(
            # nn.Linear(channels, channels // 4),
            # nn.ReLU(inplace=True),
            nn.Linear(channels, self.num_branches),
            nn.Softmax(dim=1)  #  (B, K)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, 1, T)
        B, C, _, T = x.shape

        # Global AvgPool over temporal dim
        z = x.mean(dim=-1).squeeze(2)  # shape: (B, C)

        gate_weights = self.gate(z)  # shape: (B, K)

        branch_outputs = []
        for branch in self.branches:
            out = branch(z)  # shape: (B, C)
            branch_outputs.append(out)
        # Stack to (B, K, C)
        stacked = torch.stack(branch_outputs, dim=1)  # (B, K, C)
        gate_weights = gate_weights.unsqueeze(-1)  # (B, K, 1)
        fused = (stacked * gate_weights).sum(dim=1)  # (B, C)
        scale = self.sigmoid(fused).unsqueeze(2).unsqueeze(-1)  # (B, C, 1, 1)

        return x * scale.expand_as(x)  

class DCTFAttention(nn.Module):
    def __init__(self, channels, reduction=8, fs=250, band=(8, 30)):
        super(DCTFAttention, self).__init__()
        self.fs = fs
        self.band = band
        self.reduction = reduction

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def dct_2d(self, x):
        # x: [B, C1, C2, T] -> apply 2D-DCT on last two dims (C2, T)
        # Step 1: DCT over time (T)
        N = x.size(-1)
        x_v = torch.cat([x, x.flip(dims=[-1])], dim=-1)  # [B, C1, C2, 2T]
        X = torch.fft.fft(x_v, dim=-1).real[..., :N] / 2  # [B, C1, C2, T]

        # Step 2: DCT over spatial (C2)
        N2 = x.size(-2)
        X_v = torch.cat([X, X.flip(dims=[-2])], dim=-2)  # [B, C1, 2C2, T]
        X2D = torch.fft.fft(X_v, dim=-2).real[..., :N2, :] / 2  # [B, C1, C2, T]

        return X2D

    def forward(self, x):  # x: [B, C1, C2, T]
        B, C1, C2, T = x.shape

        x_freq = self.dct_2d(x)  # [B, C1, C2, T]

        f_step = self.fs / (2 * T)
        low_idx = int(self.band[0] / f_step)
        high_idx = int(self.band[1] / f_step)

        band_power = (x_freq[..., low_idx:high_idx] ** 2).mean(dim=-1)  # [B, C1, C2]
        channel_power = band_power.mean(dim=-1)  # [B, C1]

        attn = self.fc(channel_power).unsqueeze(-1).unsqueeze(-1)  # [B, C1, 1, 1]

        return x * attn

