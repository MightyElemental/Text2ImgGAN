import torch.nn as nn
import torch
import math
from positionalencoder import PositionalEncoding

ENCODER_LAYERS = 6
ENCODER_HEADS = 10

class TransStyleGenerator(nn.Module):
    def __init__(self, ngf: int = 32, image_size: int = 64, seq_len: int = 32, enc_vec_size: int = 300) -> None:
        super(TransStyleGenerator, self).__init__()

        self.ngf = ngf
        mult = int(math.log2(image_size)) - 4

        self.seq_length = seq_len
        self.vec_size = enc_vec_size

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.vec_size, ENCODER_HEADS), ENCODER_LAYERS)

        self.pos_encoder = PositionalEncoding(self.vec_size)

        # input
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.seq_length*2, self.ngf * 2**mult, kernel_size=4, stride=2, padding=1),
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


    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """Process input(s) through the network module

        Args:
            x (Tensor): A tensor of token embeddings
        """
        # x is a list of `SEQ_LENGTH` word/tokens

        # TODO: Use checkpoints

        # TODO: Add sin/cos positional embedding

        x_encode = self.pos_encoder(x)
        x_encode = self.encoder(x_encode)
        # B * SEQ_LENGTH * ENCODER_VECTOR_SIZE
        #print(x_encode.shape)

        #z = torch.randn(B,SEQ_LENGTH, ENCODER_VECTOR_SIZE, device=z.device) # add some noise
        #print(z.shape)

        x_encode = torch.cat((x_encode, z), dim=1) # concat encoded text with noise
        #print(x_encode.shape)

        x_encode = x_encode.reshape(-1, self.seq_length*2, 4, 4)
        
        x_out = self.main(x_encode)
        
        return x_out