import torch

from utils import encode, decode

# Reading the input text and preparing the data
def prepare_data():
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.5 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, stoi, itos, vocab_size