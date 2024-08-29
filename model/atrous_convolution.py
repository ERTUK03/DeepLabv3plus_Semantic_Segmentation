import torch

class Atrous_Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Atrous_Convolution, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
