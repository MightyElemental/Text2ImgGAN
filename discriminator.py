import torch.nn as nn
import torch
from minibatchdiscriminator import MinibatchDiscriminator
from util.util import texts_to_tensor
from positionalencoder import PositionalEncoding

ENCODER_LAYERS = 6
ENCODER_HEADS = 8
ENCODER_VECTOR_SIZE = 4**2
SEQ_LENGTH = 48 # how many words in the input

class Discriminator(nn.Module):
    def __init__(self, dictionary: dict, ndf: int, image_size: int = 64):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.dictionary = dictionary
        self.image_size = image_size

        self.encode_mult = self.image_size//4

        self.embed_conv = nn.Sequential(
            nn.ConvTranspose2d(48,24,4,1,2),
            nn.ReLU(),
            nn.ConvTranspose2d(24,12,4,2,0),
            nn.ReLU(),
            nn.ConvTranspose2d(12,6,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(6,3,4,2,1),
        ) # generates shape B*3*128*128 from input B*48*4*4

        self.embed = nn.Embedding(len(dictionary), ENCODER_VECTOR_SIZE, padding_idx=dictionary["<PAD>"])
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(ENCODER_VECTOR_SIZE, ENCODER_HEADS), ENCODER_LAYERS)
        self.pos_encoder = PositionalEncoding(ENCODER_VECTOR_SIZE)

        # input
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3+3, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
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
        caps = self.pos_encoder(caps)
        caps = self.encoder(caps)
        #caps = caps.reshape(-1, SEQ_LENGTH, 4, 4)
        caps = caps.view(-1,SEQ_LENGTH,16,1).expand(-1,SEQ_LENGTH,16,16)
        #caps = caps.repeat(1,1,self.encode_mult,self.encode_mult)
        caps = self.embed_conv(caps)

        x = torch.cat((x, caps), dim=1)

        x = self.main(x)
        return x