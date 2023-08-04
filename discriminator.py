import torch.nn as nn
import torch
from minibatchdiscriminator import MinibatchDiscrimination

class Discriminator(nn.Module):
    def __init__(self, ndf: int, image_size: int = 64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        # input
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.3, inplace=True),
        )
        
        # body
        self.mult = 1
        while image_size > 4*2:
            self.add_disc_block(ndf*self.mult)
            image_size /= 2
            print(ndf*self.mult, image_size)
            self.mult *= 2

        
        self.main.add_module("conv128", nn.Sequential(
            # B * ndf*mult * 4 * 4
            nn.Conv2d(in_channels=self.ndf*self.mult, out_channels=128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.3),
            # B * 128 * 1 * 1
            nn.Flatten(),
            # B * 128
            MinibatchDiscrimination(128, 64, 3),
            nn.Flatten(),
            nn.Linear(in_features=128+64, out_features=1),
            nn.Sigmoid(),
        ))

        
    def add_disc_block(self, ndf: int):
        self.main.add_module("layer(ndf%d)"%ndf, nn.Sequential(
            nn.Conv2d(in_channels=ndf, out_channels=ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.3, inplace=True)
        ))


    def forward(self, x):
        x = self.main(x)
        return x