import torch.nn as nn
import torch
from util.util import texts_to_tensor
import math

ENCODER_LAYERS = 6
ENCODER_HEADS = 8
ENCODER_VECTOR_SIZE = 4**2
SEQ_LENGTH = 100 # how many words in the input

class TransStyleGenerator(nn.Module):
    def __init__(self, dictionary: dict, image_size: int = 64, ngf: int = 32) -> None:
        super(TransStyleGenerator, self).__init__()

        self.ngf = ngf
        mult = int(math.log2(image_size)) - 4

        self.dictionary = dictionary

        self.embed = nn.Embedding(len(dictionary), ENCODER_VECTOR_SIZE, padding_idx=dictionary["<PAD>"])
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(ENCODER_VECTOR_SIZE, ENCODER_HEADS), ENCODER_LAYERS)

        # input
        self.main = nn.Sequential(
            nn.ConvTranspose2d(SEQ_LENGTH, self.ngf * 2**mult, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.ngf* 2**mult),
            nn.ReLU(inplace=True)
        )

        # body
        while mult > 0:
            mult -= 1
            self.add_gen_block(self.ngf* 2**mult)

        # activation output
        self.main.add_module("output",nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        ))

    def add_gen_block(self, ndf: int):
        self.main.add_module("layer(ndf%d)"%ndf, nn.Sequential(
            nn.ConvTranspose2d(in_channels=ndf*2, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True)
        ))


    def forward(self, text: list[str]):
        """Process input(s) through the network module

        Args:
            x (list[str]): A list of string used as input to the network

        Returns:
            _type_: _description_
        """
        B = len(text)
        # x is a list of `SEQ_LENGTH` word/tokens

        x = texts_to_tensor(text, self.dictionary, SEQ_LENGTH)

        # TODO: Use checkpoints

        # TODO: Add sin/cos positional embedding

        x_embed = self.embed(x)
        x_encode = self.encoder(x_embed) # style input

        x_encode = x_encode.reshape(B,SEQ_LENGTH,4,4)

        x_encode += torch.randn(B,SEQ_LENGTH,4,4) # add some noise

        x_out = self.main(x_encode)
        
        return x_out