import util.webhook as webhook
from generator import TransStyleGenerator
#from discriminator import Discriminator
from tqdm import tqdm # progress bar
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from util.util import count_parameters, text_to_tensor, tokenize
import os
import torch.optim as optim
import time

IMG_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta1disc = 0.5 # Beta1 hyperparam for Adam optimizers
beta1gen = 0.5
seed = time.time()

# -= DEFINE DATA LOADERS =-

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE, antialias=True),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Normalize(0.5, 0.5)
    ])

raw_data = datasets.CocoCaptions(root="data/train2017/", annFile="data/captions_train2017.json", transform=transform)
dataloader = torch.utils.data.DataLoader(raw_data, batch_size=2, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=4)

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

gen_net = TransStyleGenerator(dictionary, IMG_SIZE)
#disc_net = Discriminator()

optimizer_gen = optim.Adam(gen_net.parameters(), lr=1e-6, betas=(beta1gen, 0.999))
#optimizer_disc = optim.Adam(disc_net.parameters(), lr=1e-6, betas=(beta1disc, 0.999))

criterion = nn.L1Loss().to(device)

print(f"embedding parameters: {count_parameters(gen_net.embed):,}")
print(f"generator parameters: {count_parameters(gen_net):,}")

# print(text_to_tensor("green alien eating cake", dictionary, 100)) # test dictionary/tensorizor

print(gen_net(["green alien eating cake"]).shape)

# -= TRAIN =-

def train(epoch:int, max_epoch:int):
    real_label = 1.0
    fake_label = 0.0
    tqRange = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch [{epoch:04d}]",
            position=1,
            leave=False,
        )
    for batch, (images,captions) in tqRange:
        optimizer_gen.zero_grad()

        print(type(captions))
        break


train(0,10)
        



