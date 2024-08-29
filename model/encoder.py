import torch
from model.modified_xception import ModifiedXception
from model.aspp import ASPP

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.backbone = ModifiedXception()
        self.aspp = ASPP(2048, 256)
        self.conv = torch.nn.Conv2d(1280, 256, 1)
        self.batchnorm = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x, x_out = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x, x_out
