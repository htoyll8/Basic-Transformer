import torch
import torch.nn as nn
from torch.nn import functional as F
from config.config import *

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # Linear transformation for keys
        self.query = nn.Linear(n_embd, head_size, bias=False) # Linear transformation for queries
        self.value = nn.Linear(n_embd, head_size, bias=False) # Linear transformation for values
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower triangular matrix for masking
        self.dropout = nn.Dropout(dropout) # Dropout layer

    def forward(self, x):
        B, T, C = x.shape # Batch size, time-steps, and channels
        k = self.key(x) # Compute keys
        q = self.query(x) # Compute queries
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # Compute attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Apply masking
        wei = F.softmax(wei, dim=-1) # Apply softmax to attention scores
        wei = self.dropout(wei) # Apply dropout
        v = self.value(x) # Compute values
        out = wei @ v # Weighted aggregation of the values
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # List of attention heads
        self.proj = nn.Linear(head_size * num_heads, n_embd) # Linear projection for concatenated heads
        self.dropout = nn.Dropout(dropout) # Dropout layer

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the output of all heads
        out = self.dropout(self.proj(out)) # Apply linear projection and dropout
        return out

class FeedForward(nn.Module):
    """ A simple feedforward layer with ReLU activation """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Linear layer
            nn.ReLU(), # ReLU activation
            nn.Linear(4 * n_embd, n_embd), # Linear layer
            nn.Dropout(dropout), # Dropout layer
        )

    def forward(self, x):
        return self.net(x) # Apply the sequential network

class Block(nn.Module):
    """ Transformer block with self-attention and feedforward layers """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # Head size
        self.sa = MultiHeadAttention(n_head, head_size) # Multi-head attention layer
        self.ffwd = FeedForward(n_embd) # Feedforward layer
        self.ln1 = nn.LayerNorm(n_embd) # Layer normalization
        self.ln2 = nn.LayerNorm(n_embd) # Layer normalization

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Self-attention with residual connection and layer normalization
        x = x + self.ffwd(self.ln2(x)) # Feedforward with residual connection and layer normalization
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Embedding layer for tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Embedding layer for positional encodings
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # Stacked transformer blocks
        self.ln_f = nn.LayerNorm(n_embd) # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) # Linear layer for predicting next token

        # Weight initialization logic (omitted in this snippet)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # Token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # Position embeddings
        x = tok_emb + pos_emb # Combine token and position embeddings
        x = self.blocks(x) # Pass through transformer blocks
        x = self.ln_f(x) # Apply final layer normalization
        logits = self.lm_head(x) # Compute logits for next token prediction

        # Loss computation logic (if targets are provided)
        loss = None
        if targets is not None:
            # Compute the loss using the logits and targets
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
