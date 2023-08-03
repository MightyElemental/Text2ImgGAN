import re
import torch

def tokenize(text: str):
    tokens = re.findall(r"([a-z']+|[.,&\-!\"$%\^£:@;#~\<\>\\\*\(\)\=\_\+\{\}\[\]\/]|\s+)", text.lower())
    tokens = [" " if t.isspace() else t for t in tokens] # replace all whitespace block with single-space
    return tokens

def pad_tokens(tokens: list[str], max_len: int):
    padded = tokens[:max_len] + ["<PAD>"]*(max_len-len(tokens))
    return padded

def text_to_tensor(text: str, dictionary: dict, max_len: int):
    tokens = tokenize(text)
    tokens = pad_tokens(tokens, max_len)
    tensor = torch.tensor([dictionary[token] for token in tokens], dtype=torch.long)
    return tensor

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)