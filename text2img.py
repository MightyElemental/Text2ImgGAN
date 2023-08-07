import util.webhook as webhook
from generator import TransStyleGenerator
from discriminator import Discriminator
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
import torchvision.utils as tvutils

IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

beta1disc = 0.5 # Beta1 hyperparam for Adam optimizers
beta1gen = 0.5
seed = time.time()

ngf = 64
ndf = 28

lr = 2e-6

current_epoch = 0

# -= DEFINE DATA LOADERS =-

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE, antialias=True),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Normalize(0.5, 0.5)
    ])

raw_data = datasets.CocoCaptions(root="data/train2017/", annFile="data/captions_train2017.json", transform=transform)
dataloader = torch.utils.data.DataLoader(raw_data, batch_size=64, shuffle=True, drop_last=True, num_workers=12,
                                          prefetch_factor=4, collate_fn=lambda batch: tuple(zip(*batch)))

print(len(raw_data))

# -= DEFINE DICTIONARY =-

dictionary = {
    "<PAD>":0
}

def load_dictionary():
    global dictionary
    if os.path.exists("./dictionary.pkl"):
        dictionary = torch.load("./dictionary.pkl")
    else:
        tqRange = tqdm(
            raw_data.ids,
            total=len(raw_data.ids),
            desc=f"dict len={len(dictionary):,}"
        )
        print("No existing dictionary found. Generating new dictionary.")
        for id in tqRange:
            caption = raw_data._load_target(id)
            for cap in caption:
                tokens = tokenize(cap)
                for tok in tokens:
                    if tok not in dictionary:
                        dictionary[tok] = len(dictionary)
            tqRange.set_description(f"dict len={len(dictionary):,}")
        print(f"Generated dictionary has {len(dictionary):,} tokens")
        torch.save(dictionary, "./dictionary.pkl")

load_dictionary()

# -= DEFINE MODELS =-

gen_net = TransStyleGenerator(dictionary, ngf, IMG_SIZE)
disc_net = Discriminator(dictionary, ndf, IMG_SIZE)

gen_net.to(device)
disc_net.to(device)

optimizer_gen = optim.Adam(gen_net.parameters(), lr=lr, betas=(beta1gen, 0.999))
optimizer_disc = optim.Adam(disc_net.parameters(), lr=lr, betas=(beta1disc, 0.999))

criterion = nn.BCELoss().to(device)

print(f"embedding parameters: {count_parameters(gen_net.embed):,}")
print(f"generator parameters: {count_parameters(gen_net):,}")
print(f"discriminator parameters: {count_parameters(disc_net):,}")

# test dictionary/tensorizor
# print(text_to_tensor("green alien eating cake", dictionary, 100))
# print(gen_net(["green alien eating cake"], device).shape)

# -= DEFINE SAVE =-

def save_checkpoint(epoch:int):
    torch.save({
        "epoch": epoch,
        "seed": seed,
        "generator": {
            "model_state": gen_net.state_dict(),
            "optimizer_state": optimizer_gen.state_dict(),
            #"loss_history": g_loss_hist,
            #"epoch_loss_history": g_epoch_loss_hist,
            "arch": {
                "ngf": gen_net.ngf
            },
        },
        "discriminator": {
            "model_state": disc_net.state_dict(),
            "optimizer_state": optimizer_disc.state_dict(),
            #"loss_history": d_loss_hist,
            #"epoch_loss_history": d_epoch_loss_hist,
            "arch": {
                "ndf": disc_net.ndf
            },
        },
    }, f"checkpoints/checkpoint-{epoch:04d}.pt")

# -= LOAD CHECKPOINT IF EXISTS =-

loaded_checkpoint = False
latest_cp = get_latest_checkpoint("checkpoints/")
ckpt_path = f"checkpoints/{latest_cp}"
if latest_cp:
    checkpoint = torch.load(ckpt_path, map_location=device)
    current_epoch = checkpoint["epoch"]+1
    seed = checkpoint["seed"]
    # generator
    gen_cp = checkpoint["generator"]
    # generator arch
    gen_arch = gen_cp["arch"]
    ngf = gen_arch["ngf"]
    # discriminator
    disc_cp = checkpoint["discriminator"]
    ndf = disc_cp["arch"]["ndf"]

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


# -= TRAIN =-

z_const = torch.randn(4, gen_net.seq_length, gen_net.vec_size, device=device)

def train_epoch(epoch: int):
    real_label = 1.0
    fake_label = 0.0
    tqRange = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch:04d}]",
            position=1,
            leave=False,
        )
    for batch, (images, captions) in tqRange:
        #images, captions = torch.tensor([i for i,c in zip(images, captions) for _ in c]), [c for cc in captions for c in cc ]
        captions = [random.choice(c) for c in captions] # select random caption from provided captions for each image
        images = torch.stack(images)
        #captions = random.choice(captions)
         
        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        # Train with batch of real images
        output = disc_net(images.to(device), captions).view(-1)
        label = torch.full((len(output),), real_label, dtype=torch.float, device=device)
        d_loss_real = criterion(output, label)
        d_loss_real.backward()

        z = torch.randn(len(output), gen_net.seq_length, gen_net.vec_size, device=device)
        # Train with batch of fake images
        fake = gen_net(captions, z)
        label.fill_(fake_label)
        output = disc_net(fake.detach(), captions).view(-1)
        d_loss_fake = criterion(output, label)
        d_loss_fake.backward()

        optimizer_disc.step()

        label.fill_(real_label)  # labels are inverted for Generator training
        output = disc_net(fake, captions).view(-1)
        g_loss = criterion(output, label)
        g_loss.backward()

        optimizer_gen.step()

        if batch % 100 == 0:
            tqRange.set_postfix_str(f"d_l_real={d_loss_real:.2}, d_l_fake={d_loss_fake:.2}, g_loss={g_loss:.2}")
            tvutils.save_image(denorm(fake), "imgs.jpg")
            tvutils.save_image(denorm(images), "real.jpg")

        

for i in range(current_epoch, current_epoch+1000):
    train_epoch(i)
    save_checkpoint(i)
    tvutils.save_image(denorm(gen_net(["alien", "A black Honda motorcycle parked in front of a garage"]*2, z_const)), f"checkpoints/alien{i}.jpg", nrow=2)

