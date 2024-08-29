import torch
from model.encoder import Encoder
from model.decoder import Decoder

class DeepLabv3(torch.nn.Module):
    def __init__(self, classes):
        super(DeepLabv3, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(classes)

    def forward(self, input):
        x, x_out = self.encoder(input)
        x = self.decoder(x, x_out)
        x = torch.nn.functional.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x
