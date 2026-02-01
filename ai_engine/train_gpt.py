import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
from gpt_torch import GPT 
from tokenizer import SimpleTokenizer 

# --- CONFIG ---
BATCH_SIZE = 32
BLOCK_SIZE = 64
MAX_ITERS = 1000
EVAL_INTERVAL = 300
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. LOAD DATA ---
def load_data(path, tokenizer):
    print(f"Stats: Loading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer.fit(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Stats: Corpus Length {len(data)} tokens. Vocab Size: {tokenizer.vocab_size}")
    return data

# --- 2. BATCH GENERATOR ---
def get_batch(data):
    # Generate a small batch of inputs x and targets y
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 3. TRAINING LOOP ---
if __name__ == "__main__":
    print(f"🚀 Training GPT from Scratch on {DEVICE}...")
    
    # Init Tokenizer
    tokenizer = SimpleTokenizer()
    dataset_path = "data/universal.txt"
    if not os.path.exists(dataset_path):
        # Fallback if running from root
        dataset_path = "ai_engine/../data/universal.txt"

    data = load_data(dataset_path, tokenizer)
    
    # Init Model
    model = GPT(vocab_size=tokenizer.vocab_size)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train
    for iter in range(MAX_ITERS):
        # Sample batch
        xb, yb = get_batch(data)

        # Evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"step {iter}: loss {loss.item():.4f}")

    print("✅ Training Complete.")
    
    # Save
    os.makedirs("models/gpt_model", exist_ok=True)
    torch.save(model.state_dict(), "models/gpt_model/weights.pth")
    tokenizer.save("models/gpt_model/vocab.json")
    print("💾 Model Saved to models/gpt_model/")
