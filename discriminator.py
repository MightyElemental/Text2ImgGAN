import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        