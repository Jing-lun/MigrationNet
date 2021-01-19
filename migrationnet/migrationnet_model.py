import torch.nn.functional as F
import torch
from .migrationnet_parts import *

class MigrationNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MigrationNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc1 = DoubleConv(64, 64)
        self.down1_1 = Down1(64, 128)
        self.down2_1 = Down1(128, 256)
        self.down3_1 = Down1(256, 512)
        self.down4_1 = Down1(512, 512)

        self.inc2 = DoubleConv(128, 128)
        self.down1_2 = Down2(128, 256)
        self.down2_2 = Down1(256, 512)
        self.down3_2 = Down1(512, 512)

        self.inc3 = DoubleConv(n_channels, 256)
        self.down1_3 = Down3(256, 512)
        self.down2_3 = Down1(512, 512)

        self.outc_global1 = OutConv(1536, 1024)
        self.outc_global2 = OutConv(1024, 512)
        self.outc         = OutConv(64, n_classes)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

    def forward(self, x):
        #64
        x_256 = x
        [a,b,c,d] = x.shape
        x_128 = x[:,0:b:2,:,:]
        x_64  = x[:,0:b:4,:,:]
        print('256',x_256.shape)
        print('128',x_128.shape)
        print('64',x_64.shape)

        #64-128-256
        x1_1 = self.inc1(x_64)
        x2_1 = self.down1_1(x1_1)
        x3_1 = self.down2_1(x2_1)
        x4_1 = self.down3_1(x3_1)
        x5_1 = self.down4_1(x4_1)

        x1_2 = self.inc2(x_128)
        x2_2 = self.down1_2(x1_2)
        x3_2 = self.down2_2(x2_2)
        x4_2 = self.down3_2(x3_2)

        x1_3 = self.inc3(x_256)
        x2_3 = self.down1_3(x1_3)
        x3_3 = self.down2_3(x2_3)


        x_sum_512 = (x4_1 + x3_2 + x2_3)
        x_sum_256 = (x3_1 + x2_2)
        x_sum_128 = x2_1

        L = [x5_1,x4_2,x3_3]
        x = torch.cat(L,1)
        x_global1 = self.outc_global1(x)
        x_global2 = self.outc_global2(x_global1)

        x = self.up1(x_sum_512, x_global2)
        x = self.up2(x, x_sum_256)
        x = self.up3(x, x_sum_128)
        x = self.up4(x, x1_1)
        logits = self.outc(x)

        return logits



if __name__ == '__main__':
    input = torch.randn(1,3,256,700)
    net = MigrationNet(n_channels=3, n_classes=1)
    output = net(input)
