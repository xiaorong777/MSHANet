import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import torch.nn as nn
from torchinfo import summary
from torchstat import stat
from utils.util import Conv2dWithConstraint, LinearWithConstraint

class spatialblock(nn.Module):
    def __init__(self, eeg_chans=22, samples=1000, spatialconv=(7,7), dropoutRate=0.5, F1=8, D=2, bias=False, n_classes=4):
        super(spatialblock, self).__init__()
        F2 = F1 * D

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=spatialconv, stride=1, padding='same', bias=bias),
            nn.BatchNorm2d(num_features=F1)
        )

        self.spatialblock = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(3,3), stride=1, padding='same', bias=bias),
            nn.BatchNorm2d(num_features=F2),
            nn.Dropout(dropoutRate)
        )

        self.pointblock = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=(1,1), stride=1, padding='same', bias=bias),
        )

        self.avgpool3 = nn.AvgPool2d(kernel_size=(3,3), stride=(3,3))

        self.chansconv = nn.Sequential(
            nn.Conv2d(in_channels=F2, out_channels=F2, kernel_size=(5,5), stride=1, padding='same', bias=bias),
            nn.BatchNorm2d(num_features=F2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.relu = nn.ReLU()
        self.avgpool5 = nn.AvgPool2d(kernel_size=(5,5), stride=(5,5))
        self.bn = nn.BatchNorm2d(num_features=F2)

    def forward(self, x):
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        y = self.block1(x)
        y3 = self.pointblock(y)
        y4 = self.avgpool3(y3)
        y1 = self.spatialblock(y) + y3
        y1 = self.avgpool3(y1)
        y2 = self.chansconv(y1) + y4
        y2 = self.relu(y2)
        y2 = self.avgpool5(y2)
        y2 = self.bn(y2)
        return y2



#%%
###============================ Initialization parameters ============================###
channels = 22
samples  = 1000

###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = spatialblock()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")
    stat(model, (1, channels, samples))
    #创建模型实例
    #model = spatialblock()

    # # 创建一个随机输入张量
    # x = torch.randn(1, 22, 1000)

    # # 获取模型的计算图
    # y = model(x)
    # dot = make_dot(y, params=dict(model.named_parameters()))

    # # 保存为PNG文件
    # dot.format = 'png'
    # dot.render("spatialblock_model")

    # 查看生成的图像
    # from PIL import Image
    # image = Image.open("spatialblock_model.png")
    # image.show()

if __name__ == "__main__":
    main()