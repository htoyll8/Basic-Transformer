import torch

from config.config import *

def encode(s, stoi):
    """
    Encodes a string into a list of integers using a given character-to-integer mapping (stoi).

    :param s: String to be encoded.
    :param stoi: Dictionary mapping characters to integer values.
    :return: List of integers representing the encoded string.
    """
    return [stoi[c] for c in s]

def decode(l, itos):
    """
    Decodes a list of integers into a string using a given integer-to-character mapping (itos).

    :param l: List of integers to be decoded.
    :param itos: Dictionary mapping integer values to characters.
    :return: String representing the decoded integers.
    """
    return ''.join([itos[i] for i in l])

# Function to generate a batch of data
def get_batch(split, data):
    # print("Data length: ", len(data), "Block size: ", block_size, "Batch: ", batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate loss
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        print("Split: ", split)
        for k in range(eval_iters):
            X, Y = get_batch(split, data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
def train_model(model, train_data, val_data):

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        # Print the current iteration number
        print(f"Iteration {iter + 1}/{max_iters}")
        print("Train data: ", train_data, "Val data: ", val_data)

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data)
        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def generate_text(model, context, max_new_tokens, stoi, itos):
    # Convert context to tensor
    context = torch.tensor(encode(context, stoi), dtype=torch.long, device=device).unsqueeze(0)
    # Generate text using the model
    generated_idx = model.generate(context, max_new_tokens)
    # Convert generated token indices to string
    generated_text = decode(generated_idx[0].tolist(), itos)
    return generated_text