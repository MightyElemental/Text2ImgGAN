import torch.nn as nn
import torch
from minibatchdiscriminator import MinibatchDiscriminator
from util.util import texts_to_tensor

ENCODER_LAYERS = 6
ENCODER_HEADS = 8
ENCODER_VECTOR_SIZE = 4**2
SEQ_LENGTH = 100 # how many words in the input

class Discriminator(nn.Module):
    def __init__(self, dictionary: dict, ndf: int, image_size: int = 64):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.dictionary = dictionary
        self.image_size = image_size

        self.embed = nn.Embedding(len(dictionary), ENCODER_VECTOR_SIZE, padding_idx=dictionary["<PAD>"])
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(ENCODER_VECTOR_SIZE, ENCODER_HEADS), ENCODER_LAYERS)

        # input
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3+SEQ_LENGTH, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.3, inplace=True),
        )
        
        # body
        self.mult = 1
        while image_size > 4*2:
            self.add_disc_block(ndf*self.mult)
            image_size /= 2
            #print(ndf*self.mult, image_size)
            self.mult *= 2

        
        self.main.add_module("conv128", nn.Sequential(
            # B * ndf*mult * 4 * 4
            nn.Conv2d(in_channels=self.ndf*self.mult, out_channels=128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.3),
            # B * 128 * 1 * 1
            nn.Flatten(),
            # B * 128
            MinibatchDiscriminator(128, 64, 3),
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


    def forward(self, x, captions):

        caps = texts_to_tensor(captions, self.dictionary, SEQ_LENGTH).to(x.device)
        caps = self.embed(caps)
        caps = self.encoder(caps)
        caps = caps.reshape(-1, 100, 4, 4)
        caps = caps.repeat(1,1,16,16)

        x = torch.cat((x, caps), dim=1)

        x = self.main(x)
        return x