#!/usr/bin/env python3

#import util.webhook as webhook
from stylegan import StyleGenerator, Discriminator, NLPLatentEncoder
from tqdm import tqdm # progress bar
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from util.util import *
import os
import torch.optim as optim
import time
import random
#import torchvision.utils as tvutils
import torchtext as tt
from math import log2
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# torch.autograd.set_detect_anomaly(True)

SEQ_LENGTH = 32 # the number of tokens the system accepts

LAMBDA_GP               = 10
Z_DIM                   = 300
W_DIM                   = 256
IN_CHANNELS             = 256
BATCH_SIZES             = [256, 128, 64, 32, 16, 8]
PROGRESSIVE_EPOCHS      = [30] * len(BATCH_SIZES)
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
CHANNELS_IMG            = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

beta1disc = 0.5 # Beta1 hyperparam for Adam optimizers
beta1gen = 0.5
seed = time.time()

lr = 1e-3

current_epoch = 0
current_step = -1

# -= LOAD EMBEDDINGS =-

embeddings = tt.vocab.GloVe(name="840B", dim=300)

tokenizer = tt.data.get_tokenizer("basic_english")

def vectorize_batch(batch):
    Y, X = zip(*batch)#
    img_size = len(Y[0][0])
    X = [tokenizer(random.choice(x)) for x in X]
    X = [tokens+[""] * (SEQ_LENGTH-len(tokens))  if len(tokens)<SEQ_LENGTH else tokens[:SEQ_LENGTH] for tokens in X]
    X_tensor = torch.zeros(len(batch), SEQ_LENGTH, Z_DIM)
    Y_tensor = torch.zeros(len(batch), 3, img_size, img_size)
    for i, tokens in enumerate(X):
        X_tensor[i] = embeddings.get_vecs_by_tokens(tokens)
    for i, img in enumerate(Y):
        Y_tensor[i] = img
    
    return X_tensor, Y_tensor

def vectorize_text(text: str):
    tokens = tokenizer(text)
    tokens = tokens+[""] * (SEQ_LENGTH-len(tokens)) if len(tokens)<SEQ_LENGTH else tokens[:SEQ_LENGTH]
    X_tensor = torch.zeros(1, SEQ_LENGTH, Z_DIM)
    X_tensor[0] = embeddings.get_vecs_by_tokens(tokens)
    return X_tensor

# -= DEFINE DATA LOADERS =-

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(IMG_SIZE, antialias=True),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.Normalize(0.5, 0.5)
#     ])

# raw_data = datasets.CocoCaptions(root="data/train2017/", annFile="data/captions_train2017.json", transform=transform)
# dataloader = DataLoader(raw_data, batch_size=64, shuffle=True, drop_last=True, num_workers=12, 
#                         prefetch_factor=4, collate_fn=vectorize_batch) # tuple(zip(*batch))

def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(image_size),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]

    raw_data = datasets.CocoCaptions(
        root="data/train2017/", 
        annFile="data/captions_train2017.json", 
        transform=transform
    )

    loader = DataLoader(
        raw_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vectorize_batch,
        num_workers=12,
        prefetch_factor=4,
    )

    print("Loaded", len(raw_data), "training items")

    return loader, raw_data


# -= DEFINE MODELS =-

nlp_encoder = NLPLatentEncoder(Z_DIM)
gen_net = StyleGenerator(Z_DIM, W_DIM, IN_CHANNELS, nlp_encoder)
disc_net = Discriminator(IN_CHANNELS, nlp_encoder)

gen_net.to(device)
disc_net.to(device)

# optimizer_gen = optim.Adam(gen_net.parameters(), lr=lr, betas=(beta1gen, 0.999))
# optimizer_disc = optim.Adam(disc_net.parameters(), lr=lr, betas=(beta1disc, 0.999))

optimizer_gen = optim.Adam([{"params": [param for name, param in gen_net.named_parameters() if "map" not in name]},
                        {"params": gen_net.map.parameters(), "lr": 1e-5}], lr=lr, betas=(0.0, 0.99))
optimizer_disc = optim.Adam(
    disc_net.parameters(), lr=lr, betas=(0.0, 0.99)
)

criterion = nn.BCELoss().to(device)

print(f"generator parameters: {count_parameters(gen_net):,}")
print(f"discriminator parameters: {count_parameters(disc_net):,}")

# test dictionary/tensorizor
# print(text_to_tensor("green alien eating cake", dictionary, 100))
# print(gen_net(["green alien eating cake"], device).shape)

# -= DEFINE SAVE =-

def save_checkpoint(step:int, seed:float):
    torch.save({
        "step": step,
        "seed": seed,
        "generator": {
            "model_state": gen_net.state_dict(),
            "optimizer_state": optimizer_gen.state_dict(),
        },
        "discriminator": {
            "model_state": disc_net.state_dict(),
            "optimizer_state": optimizer_disc.state_dict(),
        },
    }, f"checkpoints/checkpoint-{step:04d}.pt")

# -= LOAD CHECKPOINT IF EXISTS =-

loaded_checkpoint = False
latest_cp = get_latest_checkpoint("checkpoints/")
ckpt_path = f"checkpoints/{latest_cp}"
if latest_cp:
    checkpoint = torch.load(ckpt_path, map_location=device)
    current_step = checkpoint["step"]
    seed = checkpoint["seed"]
    # generator
    gen_cp = checkpoint["generator"]
    # discriminator
    disc_cp = checkpoint["discriminator"]

    loaded_checkpoint = True
    print(f"Loading checkpoint file {latest_cp}\n")
else:
    print("No existing checkpoint found. Starting new training.\n")

# Load checkpoints
if loaded_checkpoint:
    gen_net.load_state_dict(gen_cp["model_state"])
    optimizer_gen.load_state_dict(gen_cp["optimizer_state"])
    disc_net.load_state_dict(disc_cp["model_state"])
    optimizer_disc.load_state_dict(disc_cp["optimizer_state"])
    print("Loaded states")


def generate_examples(gen: StyleGenerator, steps: int, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, 32, Z_DIM).to(device)
            embeds = nlp_encoder(noise)
            img = gen(embeds, alpha, steps)
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')
            save_image(img*0.5+0.5, f"saved_examples/step{steps}/img_{i}.png")
    gen.train()

def generate_const_examples(gen: StyleGenerator, steps: int):
    gen.eval()
    gen.to(device)
    alpha = 1.0
    samples = [
        "alien",
        "A black Honda motorcycle parked in front of a garage",
        "blue shirt",
        "small, spotty elephant"
        ]
    
    encoded = [nlp_encoder(vectorize_text(x).to(device)) for x in samples]

    location = f"saved_examples/constant/step{steps}"

    if not os.path.exists(location):
        os.makedirs(location)

    for z, text in zip(encoded, samples):
        images = []
        for _ in range(16):
            img = gen(z, alpha, steps)
            img = img.squeeze(0)
            images.append(img*0.5+0.5)
        save_image(images, f"{location}/{text}.png", nrow=4)
    
    gen.train()

generate_const_examples(gen_net,1)

# -= TRAIN =-

# gradient_penalty function for WGAN-GP loss
def gradient_penalty(critic: Discriminator, real, fake, alpha: float, train_step: int, embeds, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, embeds, alpha, train_step)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train_fn(critic: Discriminator, gen: StyleGenerator, loader, dataset, step, alpha, opt_critic, opt_gen):
    loop = tqdm(loader, leave=True)

    for batch_idx, (texts, imgs) in enumerate(loop):
        texts = texts.to(device)
        imgs = imgs.to(device)
        cur_batch_size = imgs.shape[0]

        # TODO: Fix training to use images
        embeds = nlp_encoder(texts)

        #noise = torch.randn(cur_batch_size, Z_DIM).to(device)

        fake = gen(embeds, alpha, step)
        critic_real = critic(imgs, embeds, alpha, step)
        critic_fake = critic(fake.detach(), embeds, alpha, step)
        gp = gradient_penalty(critic, imgs, fake, alpha, step, embeds, device=device)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )

        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        gen_fake = critic(fake, embeds, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )


    return alpha


gen_net.train()
disc_net.train()

torch.manual_seed(seed)

# start at step that corresponds to img size that we set in config
step = int(log2(START_TRAIN_AT_IMG_SIZE / 4)) if current_step < 0 else current_step
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-5   # start with very low alpha
    loader, dataset = get_loader(4 * 2 ** step)

    print(f"Current image size: {4 * 2 ** step}")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        alpha = train_fn(
            disc_net,
            gen_net,
            loader,
            dataset,
            step,
            alpha,
            optimizer_disc,
            optimizer_gen
        )
        generate_const_examples(gen_net, step)

    save_checkpoint(step, seed)
    generate_examples(gen_net, step)
    step += 1  # progress to the next img size

