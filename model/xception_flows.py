import torch
from model.depthwise_separable_convolution import DepthwiseSeparableConv

class EntryFlow(torch.nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = torch.nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, 1, 2, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.sep_conv1 = DepthwiseSeparableConv(64, 128)
        self.sep_conv2 = DepthwiseSeparableConv(128, 128)
        self.sep_conv3 = DepthwiseSeparableConv(128, 128, 2)

        self.conv4 = torch.nn.Conv2d(128, 256, 1, 2, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.sep_conv4 = DepthwiseSeparableConv(128, 256)
        self.sep_conv5 = DepthwiseSeparableConv(256, 256)
        self.sep_conv6 = DepthwiseSeparableConv(256, 256, 2)

        self.conv5 = torch.nn.Conv2d(256, 728, 1, 2, bias=False)
        self.bn5 = torch.nn.BatchNorm2d(728)

        self.sep_conv7 = DepthwiseSeparableConv(256, 728)
        self.sep_conv8 = DepthwiseSeparableConv(728, 728)
        self.sep_conv9 = DepthwiseSeparableConv(728, 728, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x1 = self.conv3(x)
        x1 = self.bn3(x1)
        x1 = self.relu(x1)
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = x+x1
        x_out = x
        x2 = self.conv4(x1)
        x2 = self.bn4(x2)
        x2 = self.relu(x2)
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        x = x+x2
        x3 = self.conv5(x2)
        x3 = self.bn5(x3)
        x3 = self.relu(x3)
        x = self.sep_conv7(x)
        x = self.sep_conv8(x)
        x = self.sep_conv9(x)
        x = x+x3
        return x, x_out

class MiddleFlow(torch.nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.sep_conv1 = DepthwiseSeparableConv(728, 728)
        self.sep_conv2 = DepthwiseSeparableConv(728, 728)
        self.sep_conv3 = DepthwiseSeparableConv(728, 728)

    def forward(self, x):
        x1 = x
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = x+x1
        return x

class ExitFlow(torch.nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.conv = torch.nn.Conv2d(728, 1024, 1, 2, bias=False)
        self.batchnorm = torch.nn.BatchNorm2d(1024)
        self.relu = torch.nn.ReLU(inplace=True)

        self.sep_conv1 = DepthwiseSeparableConv(728, 728)
        self.sep_conv2 = DepthwiseSeparableConv(728, 1024)
        self.sep_conv3 = DepthwiseSeparableConv(1024, 1024, 2)

        self.sep_conv4 = DepthwiseSeparableConv(1024, 1536)
        self.sep_conv5 = DepthwiseSeparableConv(1536, 1536)
        self.sep_conv6 = DepthwiseSeparableConv(1536, 2048)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.batchnorm(x1)
        x1 = self.relu(x1)

        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        x = x+x1
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        return x
