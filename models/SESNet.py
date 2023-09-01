import torch
from torch import nn

from models.attention import G_SPP
from models.resnet import resnet18
from einops import rearrange

class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBNRelu, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.D3 = nn.Sequential(ConvBNRelu(512, 128), ConvBNRelu(128, 64))
        self.D2 = ConvBNRelu(128, 64)
        self.D1 = ConvBNRelu(128, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, kernel_size=1))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,s1, s2, s3):
        d3 = self.D3(s3) # 64, 32,32
        d2 = self.D2(torch.cat([self.up(d3), s2], 1))# 64,64, 64
        d1 = self.D1(torch.cat([self.up(d2), s1], 1)) # 64,128, 128
        change = self.head(self.up(d1)) # 1,128,128
        return change
class FCM(nn.Module):
    # A Feature Correction Module
    def __init__(self, e_channel, scale):
        super(FCM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.up = nn.Identity() if scale < 2 else nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.di = nn.Sequential(
            nn.Conv2d(e_channel, e_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(512, e_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel),
            nn.ReLU(inplace=True)
        )
        self.ci = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(e_channel, e_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(e_channel * 3, e_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel),
            nn.ReLU(inplace=True)
        )
        self.scale = (e_channel) ** -0.5
        self.relu = nn.ReLU()
    def forward(self, ei, r):
        r1 = self.up(self.conv(r))
        out1 = r1 * ei
        B, C, H, W = ei.shape
        di = self.di(ei).flatten(2)  # N*C*HW
        ci = self.ci(ei).flatten(2)  # N*C*49
        s = self.psi(r).flatten(2)  # N*C*49
        atten = ((ci + s) @ s.transpose(-2, -1)) * self.scale  # N*C*C
        atten = atten.softmax(dim=-1)
        si = rearrange(atten @ di, 'b c (h w) -> b c h w', h=H)
        mixed = torch.cat([si, out1, ei], 1)
        out2 = self.proj(mixed)
        out = out2
        return out
class Feature_Correction_Module(nn.Module):
    def __init__(self):
        super(Feature_Correction_Module, self).__init__()
        self.FCM1 = FCM(64, 4)
        self.FCM2 = FCM(64, 2)
        self.FCM3 = FCM(512, 1)
        self.R = G_SPP(512)
    def forward(self, e1, e2, e3):
        r = e3 # 512,32,32
        s1 = self.FCM1(e1, r) # 64, 128,128
        s2 = self.FCM2(e2, r) # 64，64，64
        s3 = self.FCM3(e3, r) # 512，32，32
        return s1, s2, s3

class FFM(nn.Module):
    # A Feature Fusion Module
    def __init__(self):
        super(FFM, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, e0_0, e0_1, e0_2, e1_0, e1_1, e1_2):
        e3 = self.up((self.conv1(torch.abs(e0_2 - e1_2))))#64,64,64
        e2 = torch.abs(e0_1 - e1_1)
        e1 = self.pool(torch.abs(e0_0 - e1_0))
        e = (e1 + e2 + e3) / 3
        e1 = self.up(e) + torch.abs(e0_0 - e1_0)
        e2 = e + torch.abs(e0_1 - e1_1)  # 64,64,64
        e3 = self.conv2(self.pool(e)) + torch.abs(e0_2 - e1_2)  # 512,32,32

        return e1, e2, e3

class SESNet(nn.Module):
    def __init__(self):
        super(SESNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.ffm = FFM()
        self.fcm = Feature_Correction_Module()
        self.Decoder = Decoder()

    def forward_single(self, x):
        x1 = self.resnet.conv1(x)# 64,128,128
        x = self.resnet.bn1(x1)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)# 64,64,64

        x2 = self.resnet.layer1(x)# 64,64,64
        x3 = self.resnet.layer2(x2)# 128,32,32
        x4 = self.resnet.layer3(x3)# 256,32，32
        x5 = self.resnet.layer4(x4)# 512,32,32
        return x1, x2, x5

    def forward(self, x1, x2):
        e0_0, e0_1, e0_2  = self.forward_single(x1)
        e1_0, e1_1, e1_2 = self.forward_single(x2)
        e1, e2, e3= self.ffm(e0_0, e0_1, e0_2, e1_0, e1_1, e1_2)
        s1, s2, s3 = self.fcm(e1, e2, e3)
        change = self.Decoder(s1, s2, s3)
        change = torch.sigmoid(change)
        return change


if __name__ == '__main__':
    x1 = torch.rand(8, 3, 512, 512)
    x2 = torch.rand(8, 3, 512, 512)
    net = SESNet()
    s1, s2, c = net(x1, x2)
    print(c.shape)


