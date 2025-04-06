"""
model.py

This module defines the network components for text-to-image generation.
It includes:
  - TransformerEncoderModule: Encodes token embeddings using a Transformer encoder,
    accepting a padding mask. It outputs a latent vector obtained via masked average pooling.
  - DCGANGenerator: A generator that upsamples a latent vector into a 256x256 image.
  - DCGANDiscriminator: A discriminator that judges whether a 256x256 image is real or generated.
  - Text2ImgModel: A wrapper that first encodes text into a latent vector with the transformer
    encoder and then generates an image using the DCGANGenerator.
    
Note: The adversarial loss and training logic are implemented outside this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from positionalencoder import PositionalEncoding


class TransformerEncoderModule(nn.Module):
    """
    TransformerEncoderModule applies a transformer encoder to a sequence of
    embedded tokens and aggregates the output to generate a latent vector.
    It accepts a padding mask to ignore padded tokens.

    Attributes:
        embed_linear (nn.Linear): Projects input embeddings to transformer dimension.
        transformer_encoder (nn.TransformerEncoder): The Transformer encoder.
        latent_proj (nn.Linear): Projects aggregated encoder output to the latent dimension.
    """
    def __init__(self, embedding_dim=300, transformer_dim=512, num_layers=4, nhead=8, dropout=0.1):
        """
        Initializes the TransformerEncoderModule.

        @param embedding_dim: Dimension of the input word embeddings.
        @param transformer_dim: Data dimension for the transformer.
        @param num_layers: Number of transformer encoder layers.
        @param nhead: Number of attention heads.
        @param dropout: Dropout rate.
        """
        super(TransformerEncoderModule, self).__init__()
        self.embed_linear = nn.Linear(embedding_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_proj = nn.Linear(transformer_dim, transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim, dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass for TransformerEncoderModule.

        @param x: Tensor of shape (batch_size, seq_length, embedding_dim).
        @param src_key_padding_mask: Optional boolean mask of shape (batch_size, seq_length)
            indicating padded elements. True means "ignore".
        @return: Latent vector of shape (batch_size, transformer_dim).
        """
        # Project to transformer dimension.
        x = self.embed_linear(x)  # (batch, seq, transformer_dim)
        # Permute for transformer encoder input: (seq, batch, features)
        x = x.permute(1, 0, 2)
        # Add positional encoding
        x = self.positional_encoding(x)
        # Pass through the transformer encoder.
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Permute back to (batch, seq, features)
        encoded = encoded.permute(1, 0, 2)
        # Use masked average pooling if a padding mask is provided.
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(2).float()  # (batch, seq, 1)
            masked_sum = torch.sum(encoded * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            aggregated = masked_sum / denom
        else:
            aggregated = torch.mean(encoded, dim=1)
        # Project aggregated vector to latent space.
        latent = self.latent_proj(aggregated)
        return latent


class DCGANGenerator(nn.Module):
    """
    DCGANGenerator generates a 256x256 image given a latent vector.
    The network first maps the latent vector via a fully-connected layer
    into a small feature map that is subsequently upsampled with a sequence
    of ConvTranspose2d layers.
    """
    def __init__(self, latent_dim=512, ngf=16, nc=3):
        """
        Initializes the generator.

        @param latent_dim: Dimension of the input latent vector.
        @param ngf: Base number of generator feature maps.
        @param nc: Number of channels in the output image.
        """
        super(DCGANGenerator, self).__init__()
        # Map latent vector to a 512x8x8 tensor.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(True)
        )
        # Upsample from 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256.
        self.net = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, ngf*16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 128x128 -> 256x256
            nn.ConvTranspose2d(ngf*2, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent):
        """
        Forward pass for the generator.

        @param latent: Tensor of shape (batch_size, latent_dim).
        @return: Generated image tensor of shape (batch_size, nc, 256, 256).
        """
        x = self.fc(latent)
        x = x.view(latent.size(0), 512, 8, 8)
        img = self.net(x)
        return img


class DCGANDiscriminator(nn.Module):
    """
    DCGANDiscriminator determines whether a given 256x256 image is real or fake.
    It downsamples the input image with a series of convolutional layers
    and produces a scalar output.
    """
    def __init__(self, nc=3, ndf=16):
        """
        Initializes the discriminator.

        @param nc: Number of channels in the input image.
        @param ndf: Base number of discriminator feature maps.
        """
        super(DCGANDiscriminator, self).__init__()
        # Downsample: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4.
        self.net = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128x128 -> 64x64
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 -> 32x32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 16, ndf * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Final fully-connected layer.
        self.fc = nn.Linear(ndf * 32 * 4 * 4, 1)

    def forward(self, img):
        """
        Forward pass for the discriminator.

        @param img: Input image tensor of shape (batch_size, nc, 256, 256).
        @return: Tensor of shape (batch_size, 1) with discriminator logits.
        """
        x = self.net(img)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class Text2ImgModel(nn.Module):
    """
    Text2ImgModel integrates the text encoder and the image generator.
    The model accepts text embeddings (and corresponding padding masks) and outputs
    a latent vector; the DCGANGenerator then transforms this latent vector into
    a 256x256 image.
    """
    def __init__(self, embedding_dim=300, transformer_dim=512, num_layers=4, nhead=8,
                 dropout=0.1, nc=3):
        """
        Initializes the text-to-image model.

        @param embedding_dim: Dimension of input token embeddings.
        @param transformer_dim: Hidden dimension for transformer and latent space.
        @param num_layers: Number of transformer encoder layers.
        @param nhead: Number of attention heads in the transformer.
        @param dropout: Dropout rate for the transformer.
        @param nc: Number of channels in the generated image.
        """
        super(Text2ImgModel, self).__init__()
        self.encoder = TransformerEncoderModule(embedding_dim, transformer_dim, num_layers, nhead, dropout)
        self.generator = DCGANGenerator(latent_dim=transformer_dim, nc=nc)

    def forward(self, text_embeddings, src_key_padding_mask=None):
        """
        Forward pass for Text2ImgModel.

        @param text_embeddings: Tensor of shape (batch_size, seq_length, embedding_dim).
        @param src_key_padding_mask: Boolean tensor of shape (batch_size, seq_length)
            indicating which tokens are padding.
        @return: Generated images of shape (batch_size, nc, 256, 256).
        """
        latent = self.encoder(text_embeddings, src_key_padding_mask)
        generated_img = self.generator(latent)
        return generated_img
