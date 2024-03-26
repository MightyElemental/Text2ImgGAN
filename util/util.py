import re
import torch
import torchvision.transforms as transforms
import os
import torchtext as tt

tokenizer = tt.data.get_tokenizer("basic_english")

def tokenize(text: str):
    return tokenizer(text)

def pad_tokens(tokens: list[str], max_len: int):
    padded = tokens[:max_len] + [""]*(max_len-len(tokens))
    return padded

def text_to_tensor(text: str, dictionary: dict, max_len: int):
    tokens = tokenize(text)
    tokens = pad_tokens(tokens, max_len)
    tensor = torch.tensor([dictionary[token] for token in tokens], dtype=torch.long).reshape(1,len(tokens))
    return tensor

def texts_to_tensor(texts: list[str], dictionary: dict, max_len: int):
    return torch.cat([text_to_tensor(text, dictionary, max_len) for text in texts], dim=0)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

denormalize = transforms.Normalize(-1, 2)

def denorm(x: torch.tensor):
    x = x.squeeze(0)
    x = denormalize(x)
    return torch.clamp(x, 0, 1)

def get_latest_checkpoint(path:str)->str:
    if not os.path.isdir(path): return None
    files = os.listdir(path)
    # only accept checkpoint*.pt files
    files = [f for f in files if f.endswith(".pt") and f.startswith("checkpoint")]
    return max(files, key=file_num) if files else None

def file_num(f:str):
    n = re.findall("e(\d{4,})",f)
    return int(n[0]) if n else -1, f

def make_path_if_not_exist(path:str):
    if not os.path.isdir(path):
        os.makedirs(path)