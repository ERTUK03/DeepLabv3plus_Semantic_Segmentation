import torch

class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, 3, stride, groups=in_channels, padding=1, bias=False)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
