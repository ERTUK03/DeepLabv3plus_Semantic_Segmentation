import torch
from model.atrous_convolution import Atrous_Convolution

class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_conv1 = Atrous_Convolution(in_channels, out_channels, 1, 1)
        self.atrous_conv2 = Atrous_Convolution(in_channels, out_channels, 3, 6)
        self.atrous_conv3 = Atrous_Convolution(in_channels, out_channels, 3, 12)
        self.atrous_conv4 = Atrous_Convolution(in_channels, out_channels, 3, 18)
        self.global_avg_pool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                                   torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                                   torch.nn.BatchNorm2d(out_channels),
                                                   torch.nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.atrous_conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x4 = self.atrous_conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = torch.nn.functional.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return x
