import numpy as np
try:
    import cupy as cp
    USING_GPU = True
    np = cp # Alias cupy as np for the class
    import numpy as real_numpy
except ImportError:
    USING_GPU = False
    import numpy as real_numpy

import os
import time
import json
import sys
from tokenizer import SimpleTokenizer

# DLL Patch (Standard)
patch_dir = os.path.abspath("dll_patch")
if os.path.exists(patch_dir):
    try:
        os.add_dll_directory(patch_dir)
        os.environ['PATH'] = patch_dir + ";" + os.environ['PATH']
    except: pass

class TokenGRUTrainer:
    def __init__(self, hidden_size=256, embedding_size=256, vocab_limit=3000, seq_length=20, output_dir="./models/token_model"):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.seq_length = seq_length # Shorter seq length for tokens (words contain more info than chars)
        self.learning_rate = 1e-2
        self.output_dir = output_dir
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_limit)
        
        # Weights
        self.We = None # Embeddings
        self.Wz, self.Wr, self.Wh = None, None, None
        self.bz, self.br, self.bh = None, None, None
        self.Why = None; self.by = None

    def load_data(self, dataset_path):
        print(f"[Data] Reading {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if len(text) < 10: 
            print("⚠️ Data file seems empty!")
            return []

        print(f"[Data] Raw text size: {len(text)} chars. Tokenizing...")
        self.tokenizer.fit(text)
        tokens = self.tokenizer.encode(text)
        print(f"[Data] Tokenized sequence length: {len(tokens)}")
        return tokens

    def init_model(self):
        vocab_size = self.tokenizer.vocab_size
        
        def xavier(r, c):
            return np.random.randn(r, c) * np.sqrt(2.0/(r+c))
            
        # Embedding Matrix (Vocab x Embed)
        self.We = xavier(vocab_size, self.embedding_size)
        
        # GRU Weights
        # Input is now Embedding Vector
        input_dim = self.embedding_size + self.hidden_size
        
        self.Wz = xavier(self.hidden_size, input_dim)
        self.Wr = xavier(self.hidden_size, input_dim)
        self.Wh = xavier(self.hidden_size, input_dim)
        
        self.bz = np.zeros((self.hidden_size, 1))
        self.br = np.zeros((self.hidden_size, 1))
        self.bh = np.zeros((self.hidden_size, 1))
        
        # Output
        self.Why = xavier(vocab_size, self.hidden_size)
        self.by = np.zeros((vocab_size, 1))
        
        print(f"[Init] Token GRU Initialized on { 'GPU' if USING_GPU else 'CPU' }.")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, dataset_path, iterations=20000):
        data = self.load_data(dataset_path)
        self.init_model()
        
        # Adagrad Memory
        mWe = np.zeros_like(self.We)
        mWz, mWr, mWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
        mbz, mbr, mbh = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh)
        mWhy, mby = np.zeros_like(self.Why), np.zeros_like(self.by)
        
        vocab_size = self.tokenizer.vocab_size
        smooth_loss = -np.log(1.0/vocab_size)*self.seq_length
        
        p = 0
        hprev = np.zeros((self.hidden_size, 1))
        
        print(f"[Train] Starting Token-Level Training ({iterations} steps)...")
        
        for i in range(iterations):
            if p + self.seq_length + 1 >= len(data):
                p = 0
                hprev = np.zeros((self.hidden_size, 1))
                
            inputs_ix = data[p:p+self.seq_length]
            targets_ix = data[p+1:p+self.seq_length+1]
            
            # --- Forward ---
            xs, hs, zs, rs, h_hats = {}, {}, {}, {}, {}
            hs[-1] = np.copy(hprev)
            loss = 0
            ps = {}
            
            for t in range(len(inputs_ix)):
                # Embedding Lookup
                # x is (Embed_Size, 1)
                ix = inputs_ix[t]
                xs[t] = self.We[ix].reshape(-1, 1) 
                
                # GRU
                combined = np.vstack((xs[t], hs[t-1]))
                zs[t] = self.sigmoid(np.dot(self.Wz, combined) + self.bz)
                rs[t] = self.sigmoid(np.dot(self.Wr, combined) + self.br)
                combined_r = np.vstack((xs[t], rs[t] * hs[t-1]))
                h_hats[t] = np.tanh(np.dot(self.Wh, combined_r) + self.bh)
                hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_hats[t]
                
                ys = np.dot(self.Why, hs[t]) + self.by
                ps[t] = np.exp(ys) / np.sum(np.exp(ys))
                loss += -np.log(ps[t][targets_ix[t], 0])
                
            # --- Backward ---
            dWe = np.zeros_like(self.We) # Sparse updates ideally, but dense here for simplicity
            dWz, dWr, dWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
            dWhy = np.zeros_like(self.Why)
            dbz, dbr, dbh = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])
            
            for t in reversed(range(len(inputs_ix))):
                dy = np.copy(ps[t])
                dy[targets_ix[t]] -= 1
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                
                dh = np.dot(self.Why.T, dy) + dh_next
                
                # GRU Backprop
                d_h_hat = dh * zs[t] * (1 - h_hats[t]**2)
                d_z = dh * (h_hats[t] - hs[t-1]) * zs[t] * (1 - zs[t])
                
                combined_r = np.vstack((xs[t], rs[t] * hs[t-1]))
                dWh += np.dot(d_h_hat, combined_r.T)
                dbh += d_h_hat
                # Input grad specific to Wh
                dx_h = np.dot(self.Wh[:, :self.embedding_size].T, d_h_hat)
                
                combined = np.vstack((xs[t], hs[t-1]))
                dWz += np.dot(d_z, combined.T)
                dbz += d_z
                # Input grad specific to Wz
                dx_z = np.dot(self.Wz[:, :self.embedding_size].T, d_z)
                
                # Reset Gate
                dr = np.dot(self.Wh[:, self.embedding_size:].T, d_h_hat) * hs[t-1] * rs[t] * (1 - rs[t])
                dWr += np.dot(dr, combined.T)
                dbr += dr
                # Input grad specific to Wr
                dx_r = np.dot(self.Wr[:, :self.embedding_size].T, dr)
                
                # Accumulate Embedding Gradient
                dx = dx_h + dx_z + dx_r
                # Update the specific row in dWe corresponding to input token
                # In Cupy/Numpy this is a bit tricky to do purely sparse in this loop structure
                # We'll just scatter add
                ix = inputs_ix[t]
                dWe[ix] += dx.ravel()
                
                dh_next = dh * (1 - zs[t]) # Simplified flow
                
            # Clip
            for d in [dWe, dWz, dWr, dWh, dWhy]:
                np.clip(d, -5, 5, out=d)
                
            # Update (Adagrad)
            # Embedding Update
            mWe += dWe**2
            self.We += -self.learning_rate * dWe / np.sqrt(mWe + 1e-8)
            
            for param, dparam, mem in zip([self.Wz, self.Wr, self.Wh, self.Why, self.bz, self.br, self.bh, self.by],
                                          [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby],
                                          [mWz, mWr, mWh, mWhy, mbz, mbr, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
                
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            p += self.seq_length
            hprev = hs[len(inputs_ix)-1]
            
            if i % 100 == 0:
                print(f"Step {i}/{iterations} | Loss: {smooth_loss:.4f}")
        
        self.save_model()

    def save_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Save Tokenizer
        self.tokenizer.save(f"{self.output_dir}/vocab.json")
        
        def cpu(x):
            try: return real_numpy.array(x.get())
            except: return x
            
        real_numpy.savez(f"{self.output_dir}/weights.npz",
            We=cpu(self.We),
            Wz=cpu(self.Wz), Wr=cpu(self.Wr), Wh=cpu(self.Wh), Why=cpu(self.Why),
            bz=cpu(self.bz), br=cpu(self.br), bh=cpu(self.bh), by=cpu(self.by),
            architecture="TOKEN_GRU"
        )
        print("[Save] Token-Level Model Saved.")

if __name__ == "__main__":
    trainer = TokenGRUTrainer(vocab_limit=5000)
    # Point to the new plain text file we just created
    trainer.train("data/training_data.txt", iterations=100) # Only 100 for local test (Speed)
