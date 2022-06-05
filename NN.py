from tokenize import Double
import torch.nn as nn
import torch
import torch.nn.functional as F
from zmq import device
import numpy as np
from torch.nn.functional import conv2d
from createdata import step, BB
from torchvision.models import resnet18

PTS = torch.from_numpy(BB).float()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(

            nn.Conv3d(in_channels, mid_channels, kernel_size=(3,3,3), padding=padding, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=(3,3,3), padding=padding, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3,3,3), padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()

        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels//4, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels//4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels//4)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels//4)
        )

        self.layer4 = nn.Sequential(
            nn.Conv3d(out_channels//4, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        shortcut = self.downsample(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        x = x + shortcut
        return torch.relu(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2,2),
            ResBottleneckBlock(in_channels, out_channels)
        )
    
    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = ResBottleneckBlock(in_channels, out_channels)    

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,  diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1))

    def forward(self, x):
        return self.conv(x)

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.linear_layer(x)


class NeuralNet(nn.Module):
    def __init__(self, device="cuda"):
        super(NeuralNet, self).__init__()
        self.device = device

        # Unet
        self.inc = ResBottleneckBlock(4,32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512) 
        self.d5 = Down(512,512)

        self.u1 = Up(512+512, 512)
        self.u2 = Up(512+256, 256)
        self.u3 = Up(256+128, 128)
        self.u4 = Up(128+64, 64)
        self.u5 = Up(64+32,32)
        self.out = OutConv(32, 1)
        
    def forward(self, sdf):    # [B,64,64,64] 
        B = sdf.shape[0]
        dim = sdf.shape[-1]
        sdf = sdf.view(B,1,dim,dim,dim)                             # [B,1,dim,dim,dim]
        pts = PTS.to(self.device).unsqueeze(0).expand(B,-1,-1)      # [B,dim*dim*dim,3]
        pts = pts.view(B,dim,dim,dim,3).permute(0,4,1,2,3)          # [B,3,dim,dim,dim]
        x = torch.cat((sdf,pts), dim=1)                             # [B,4,dim,dim,dim]
  
        d = self.inc(x)         # [B,64,64,64,64]
        d1 = self.d1(d)         # [B,128,32,32,32]
        d2 = self.d2(d1)        # [B,256,16,16,16]
        d3 = self.d3(d2)        # [B,512,8,8,8]
        d4 = self.d4(d3)        # [B,1024,4,4,4]
        d5 = self.d5(d4)        # [B,1024,2,2,2]

        u1 = self.u1(d5,d4)     # [B,1024,4,4,4]
        u2 = self.u2(u1,d3)     # [B,512,8,8,8]
        u3 = self.u3(u2,d2)     # [B,256,16,16,16]
        u4 = self.u4(u3,d1)     # [B,128,32,32,32]
        u5 = self.u5(u4,d)      # [B,64,64,64,64]
        df_pred = self.out(u5)  # [B,1,64,64,64]

        df_pred = df_pred.view(B,dim,dim,dim)
    
        return df_pred 