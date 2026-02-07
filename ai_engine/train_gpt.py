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
MAX_ITERS = 50000
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
    tokenizer = SimpleTokenizer(vocab_size=10000)
    dataset_path = "data/universal_v2.txt"
    if not os.path.exists(dataset_path):
        # Fallback if running from root
        dataset_path = "ai_engine/../data/universal_v2.txt"

    data = load_data(dataset_path, tokenizer)
    
    # Init Model
    model = GPT(vocab_size=tokenizer.vocab_size)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # CHECKPOINTING
    start_iter = 0
    checkpoint_path = "models/gpt_model/checkpoint.pth"
    os.makedirs("models/gpt_model", exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter']
        print(f"   -> Resuming at Step {start_iter}")

    # Train
    for iter in range(start_iter, MAX_ITERS):
        # Sample batch
        xb, yb = get_batch(data)

        # Evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"step {iter}: loss {loss.item():.4f}")
            
        # AUTO-SAVE every 500 steps
        if iter > 0 and iter % 500 == 0:
            print(f"💾 Saving Checkpoint at step {iter}...")
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            # Also save final weights for inference
            torch.save(model.state_dict(), "models/gpt_model/weights.pth")
            tokenizer.save("models/gpt_model/vocab.json")

    print("✅ Training Complete.")
    
    # Final Save
    torch.save(model.state_dict(), "models/gpt_model/weights.pth")
    tokenizer.save("models/gpt_model/vocab.json")
    # Remove checkpoint to start fresh next time (optional, but good for "Clean" release)
    # os.remove(checkpoint_path) 
    print("💾 Model Saved to models/gpt_model/")
