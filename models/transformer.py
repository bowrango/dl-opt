# Transformers

import torch 
import torch.nn.functional as F
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(2025)

n_embd = 32
num_heads = 4
head_size = 8
blk_size = 32

class Head(nn.Module):
    def __init__(self, head_size=head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blk_size, blk_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention score
        # (B,T,C) @ (B,C,T) -> (B,T,T)
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T,:T]==0, float('-inf')) # Decoder
        w = F.softmax(w, dim=-1)
        v = self.value(x) # (B,T,C)
        # (B,T,C) <- (B,T,T) @ (B,T,C)
        out = w @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=num_heads, head_size=head_size):
        super().__init__()
        assert(head_size//num_heads==0)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], -1)
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-normalization with residual connections
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd=32, num_heads=4, num_layers=4, blk_size=32, dropout=0.1):
        super().__init__()
        self.blk_size = blk_size

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(blk_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Generate embeddings
        tok_emb = self.token_embedding(x)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context to blk_size
            idx_cond = idx[:, -self.blk_size:]

            # Focus on last time step
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
