import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CONFIGURATION ---
# This defines how "Big" and "Complex" your brain is.
class GPTConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.embed_dim = 256      # Size of each word vector
        self.num_heads = 4        # How many "concepts" it can focus on at once
        self.num_layers = 4       # How deep the reasoning is
        self.block_size = 64      # Max sentence length it can remember
        self.dropout = 0.1

# --- 1. SELF-ATTENTION (The "Smart" Part) ---
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.embed_dim, cfg.embed_dim // cfg.num_heads, bias=False)
        self.query = nn.Linear(cfg.embed_dim, cfg.embed_dim // cfg.num_heads, bias=False)
        self.value = nn.Linear(cfg.embed_dim, cfg.embed_dim // cfg.num_heads, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        # Compute attention scores ("affinity")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Decoder masking (Can't see future)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.embed_dim // cfg.num_heads
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.num_heads)])
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

# --- 2. FEED FORWARD (The "Thinking" Part) ---
class FeedFoward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- 3. TRANSFORMER BLOCK ---
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedFoward(cfg)
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # Residual connections (x + ...)
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. THE FULL GPT MODEL ---
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        cfg = GPTConfig(vocab_size)
        self.config = cfg
        
        # Embedding Table (Words -> Vectors)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        # Position Embedding (Order of words matters)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.embed_dim)
        
        # Transformer Blocks (Layers of Thinking)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.num_layers)])
        
        # Final Norm & Output
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Get Token Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        # 2. Get Position Embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        
        # 3. Combine
        x = tok_emb + pos_emb # (B,T,C)
        
        # 4. Process through Transformer
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)   # (B,T,C)
        
        # 5. Output Logits (Guess next word)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
