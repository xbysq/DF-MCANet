import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x1):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x1 * self.sigmoid(out)
        return out
class SPModel(nn.Module):
    def __init__(self, ch):
        super(SPModel, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.fc1 = nn.Conv2d(in_channels=ch, out_channels=ch//16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(ch//16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=ch//16, out_channels=ch, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = self.bn1(self.fc1(x))
        x = self.fc2(self.relu(x))
        x = self.bn2(x)
        out = self.relu(x + identity)
        return out
class G_SPP(nn.Module):
    # Global space pyramid pooling
    def __init__(self, c1, k=5, scale=1.0):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 =nn.Sequential(nn.Conv2d(c1, c_, 1, 1),
                                nn.BatchNorm2d(c_),
                                nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv2d(c_, c1, 1, 1),
                                 nn.BatchNorm2d(c1))
        self.cv3 = nn.Sequential(nn.Conv2d(c_ * 4, c1, 1, 1),
                                 nn.BatchNorm2d(c1),
                                 nn.ReLU())
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.FloatTensor([scale]), requires_grad=True)
    def forward(self, x):
        x = self.cv1(x) # H*W*C/2
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        atten = self.sigmoid(self.cv2(self.pool(x+y1+y2+y3)) * self.scale)# 256,1,1
        return atten * self.cv3(torch.cat((x, y1, y2, y3), 1))
