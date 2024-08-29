import torch

class Decoder(torch.nn.Module):
    def __init__(self, classes):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(128, 48, 1)
        self.batchnorm = torch.nn.BatchNorm2d(48)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(304, classes, 3, padding=1)

    def forward(self, x, x_out):
        x_out = self.conv1(x_out)
        x_out = self.batchnorm(x_out)
        x_out = self.relu(x_out)
        x = torch.nn.functional.interpolate(x, size=x_out.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x_out), dim=1)
        x = self.conv2(x)
        return x
