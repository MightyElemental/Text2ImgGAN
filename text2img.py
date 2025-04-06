#!/usr/bin/env python3

"""
text2img.py

This script trains a text-to-image model that generates 256x256 images from input text.
It loads the COCO Captions dataset with transforms v2 (Resize, RandomHorizontalFlip,
CenterCrop, and Normalize), tokenizes the captions, and converts them into embeddings
using pre-trained GloVe. Captions are padded/truncated to 64 tokens; a corresponding mask
is generated to ignore padded tokens in the transformer encoder.

The generator network uses a transformer encoder to process text and produce a latent vector.
That latent vector is passed to a DCGAN generator that outputs a 256x256 image.
A DCGAN discriminator is trained adversarially against the generator.
Model checkpoints are saved after each epoch, and training statistics are logged to TensorBoard.
Additionally, every 500 batches eight generated images (using eight fixed text prompts)
are sent to TensorBoard.

Usage:
    python text2img.py --epochs 10 --batch_size 32 --learning_rate 0.0002
"""

import os
import argparse
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import CocoCaptions
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

# Import the model definitions.
from model import Text2ImgModel, DCGANDiscriminator

# -----------------------------------------------------------------------------
# Define data transforms using torchvision.transforms.v2.
# -----------------------------------------------------------------------------
transform = v2.Compose([
    v2.Resize(256),
    v2.RandomHorizontalFlip(),
    v2.CenterCrop(256),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# Data Processing Utilities.
# -----------------------------------------------------------------------------
def process_caption(caption, tokenizer, embeddings, max_length=64):
    """
    Process a single caption: tokenize and convert each token to a vector using
    pre-trained embeddings. Pads/truncates to max_length and generates a corresponding
    padding mask (True for padded positions).

    @param caption: Caption string.
    @param tokenizer: Callable that tokenizes a string.
    @param embeddings: Pre-trained embeddings (e.g., GloVe).
    @param max_length: Maximum token sequence length.
    @return: Tuple (tensor, mask) where tensor is shape (max_length, embedding_dim)
             and mask is a boolean tensor of shape (max_length,) with True in padded positions.
    """
    tokens = tokenizer(caption.lower())
    embedding_dim = embeddings.dim
    tensor = torch.zeros(max_length, embedding_dim)
    mask = torch.ones(max_length, dtype=torch.bool)  # True indicates a padded token.
    for i in range(min(len(tokens), max_length)):
        token = tokens[i]
        if token in embeddings.stoi:
            tensor[i] = embeddings.vectors[embeddings.stoi[token]]
        else:
            tensor[i] = torch.zeros(embedding_dim)
        mask[i] = False  # Valid (non-padded) token.
    return tensor, mask


def collate_fn(batch, tokenizer, embeddings, max_length=64):
    """
    Custom collate function for the dataloader. Processes a batch and returns:
     - images (tensor)
     - text embeddings (tensor)
     - padding masks (tensor)
    
    @param batch: List of tuples (image, captions).
    @param tokenizer: Text tokenizer.
    @param embeddings: Pre-trained GloVe embeddings.
    @param max_length: Maximum token sequence length.
    @return: Tuple (batch_images, batch_text_embeddings, batch_masks)
    """
    images = []
    text_embeddings = []
    masks = []
    for img, captions in batch:
        images.append(img)
        if captions:
            emb, m = process_caption(captions[0], tokenizer, embeddings, max_length)
        else:
            emb = torch.zeros(max_length, embeddings.dim)
            m = torch.ones(max_length, dtype=torch.bool)
        text_embeddings.append(emb)
        masks.append(m)
    images = torch.stack(images)                 # (batch, channels, H, W)
    text_embeddings = torch.stack(text_embeddings)  # (batch, max_length, embedding_dim)
    masks = torch.stack(masks)                     # (batch, max_length)
    return images, text_embeddings, masks

# -----------------------------------------------------------------------------
# Fixed text prompts for TensorBoard logging.
# -----------------------------------------------------------------------------
FIXED_PROMPTS = [
    "a small dog in the park",
    "a red sports car driving fast",
    "a delicious plate of pasta on the table",
    "a scenic mountain landscape during sunrise",
    "a futuristic city skyline at night",
    "a closeup of a blooming flower",
    "a cat sitting on a window sill",
    "a group of people dancing at a festival"
]

def generate_fixed_examples(model, tokenizer, embeddings, device):
    """
    Generate images for eight fixed text prompts for logging to TensorBoard.
    
    @param model: The Text2ImgModel.
    @param tokenizer: Tokenizer for text.
    @param embeddings: Pre-trained embeddings.
    @param device: Torch device.
    @return: Generated images tensor of shape (8, 3, 256, 256).
    """
    model.eval()
    texts = []
    masks = []
    for prompt in FIXED_PROMPTS:
        emb, m = process_caption(prompt, tokenizer, embeddings, max_length=64)
        texts.append(emb)
        masks.append(m)
    text_embeddings = torch.stack(texts).to(device)  # (8, 64, embedding_dim)
    masks = torch.stack(masks).to(device)              # (8, 64)
    with torch.no_grad():
        fake_imgs = model(text_embeddings, src_key_padding_mask=masks)
    model.train()
    return fake_imgs

def weights_init_normal(m):
    """
    Initialize model weights with a normal distribution.
    Applies to Conv2d, ConvTranspose2d, Linear, and BatchNorm2d layers.
    
    @param m: Module layer to initialize.
    """
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


# -----------------------------------------------------------------------------
# Training function implementing adversarial training.
# -----------------------------------------------------------------------------
def train(args):
    """
    Train the text-to-image model with adversarial training.
    Uses a generator (Text2ImgModel) and a discriminator (DCGANDiscriminator).
    
    @param args: Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up TensorBoard writer.
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)

    # Initialize dataset and dataloader.
    dataset = CocoCaptions(
        root="data/train2017/",
        annFile="data/captions_train2017.json",
        transform=transform
    )
    tokenizer = get_tokenizer("basic_english")
    embeddings = torchtext.vocab.GloVe(name="840B", dim=300)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, embeddings, max_length=64),
        num_workers=20
    )

    # Instantiate models.
    model = Text2ImgModel(
        embedding_dim=300,
        transformer_dim=512,
        num_layers=4,
        nhead=8,
        dropout=0.1,
        nc=3
    ).to(device)
    discriminator = DCGANDiscriminator(nc=3, ndf=64).to(device)

    # After instantiating your models, initialize their weights:
    # For example, if 'model' is your generator and 'discriminator' is your DCGANDiscriminator:
    model.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Optimizers for generator and discriminator.
    optimizer_G = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Use BCEWithLogitsLoss.
    criterion = nn.BCEWithLogitsLoss()
    
    # Use one-sided label smoothing for real images.
    real_label = 0.9
    fake_label = 0.0

    global_step = 0
    # Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch_idx, (images, text_embeddings, masks) in enumerate(epoch_iterator):
            batch_size_curr = images.size(0)
            images = images.to(device)  # Real images (already 256x256 from the transforms).
            text_embeddings = text_embeddings.to(device)
            masks = masks.to(device)
            real_imgs = images  # No resizing needed.

            # Create labels.
            real_labels = torch.full((batch_size_curr, 1), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size_curr, 1), fake_label, dtype=torch.float, device=device)
            
            # -------------------------------------------------------
            # 1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))).
            # -------------------------------------------------------
            discriminator.zero_grad()
            # Discriminator loss on real images.
            output_real = discriminator(real_imgs)
            loss_D_real = criterion(output_real, real_labels)
            # Generate fake images.
            fake_imgs = model(text_embeddings, src_key_padding_mask=masks)
            output_fake = discriminator(fake_imgs.detach())
            loss_D_fake = criterion(output_fake, fake_labels)
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            
            # -------------------------------------------------------
            # 2) Update Generator: maximize log(D(G(z))).
            # -------------------------------------------------------
            model.zero_grad()
            output_fake_for_G = discriminator(fake_imgs)
            loss_G = criterion(output_fake_for_G, real_labels)
            loss_G.backward()
            optimizer_G.step()
            
            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()
            avg_loss_D = running_loss_D / (batch_idx + 1)
            avg_loss_G = running_loss_G / (batch_idx + 1)
            
            epoch_iterator.set_postfix(loss_D=avg_loss_D, loss_G=avg_loss_G)
            writer.add_scalar("Loss/Discriminator", loss_D.item(), global_step)
            writer.add_scalar("Loss/Generator", loss_G.item(), global_step)

            # Every 500 batches log eight example images (from fixed text prompts) to TensorBoard.
            if global_step % 250 == 0:
                fake_examples = generate_fixed_examples(model, tokenizer, embeddings, device)
                # fake_examples shape: (8, 3, 256, 256), scale to [0,1] from [-1,1]
                fake_examples = (fake_examples + 1) / 2.0
                writer.add_images(tag="FixedExamples", img_tensor=fake_examples, global_step=global_step)

            global_step += 1
    
        # Save checkpoints after each epoch.
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        model_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pth")
        disc_path = os.path.join(args.checkpoint_dir, f"discriminator_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(discriminator.state_dict(), disc_path)
        print(f"Epoch {epoch}: Generator checkpoint saved to {model_path}")
        print(f"Epoch {epoch}: Discriminator checkpoint saved to {disc_path}")

    writer.close()


def parse_args():
    """
    Parse command-line arguments.
    
    @return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train text-to-image model using transformer encoder and DCGAN at 256x256 resolution.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
