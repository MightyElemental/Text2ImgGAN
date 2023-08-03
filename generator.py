import torch.nn as nn
import torch
from util.util import text_to_tensor

ENCODER_LAYERS = 6
ENCODER_HEADS = 8
ENCODER_VECTOR_SIZE = 16
SEQ_LENGTH = 100 # how many words in the input

class TransStyleGenerator(nn.Module):
    def __init__(self, dictionary: dict) -> None:
        super().__init__()

        self.dictionary = dictionary

        self.embed = nn.Embedding(len(dictionary), ENCODER_VECTOR_SIZE, padding_idx=dictionary["<PAD>"])
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(ENCODER_VECTOR_SIZE, ENCODER_HEADS), ENCODER_LAYERS)


    def forward(self, text: str, seed: int):
        """Process input(s) through the network module

        Args:
            x (str): A string used as input to the network
            seed (int): The seed of the generator

        Returns:
            _type_: _description_
        """
        # x is a list of `SEQ_LENGTH` word/tokens

        x = text_to_tensor(text, self.dictionary, SEQ_LENGTH)

        # TODO: Use checkpoints

        # TODO: Add sin/cos positional embedding

        x_embed = self.embed(x)
        x_encode = self.encoder(x_embed)

        # TODO: Pass x_encode into stylegan
        
        return x_encode