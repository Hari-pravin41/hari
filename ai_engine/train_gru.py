import numpy as np
try:
    import cupy as cp
    USING_GPU = True
    np = cp # Alias cupy as np for the class
    import numpy as real_numpy # Keep real numpy for saving
except ImportError:
    USING_GPU = False
    import numpy as real_numpy

import os
import time
import json
import sys

# ADD CUDA TO PATH MANUALLY (Professional Fix)
# (Re-using the fix from train_scratch.py)
patch_dir = os.path.abspath("dll_patch")
if os.path.exists(patch_dir):
    try:
        os.add_dll_directory(patch_dir)
        os.environ['PATH'] = patch_dir + ";" + os.environ['PATH']
    except Exception: pass

class GRUTrainer:
    def __init__(self, hidden_size=256, seq_length=50, output_dir="./models/numpy_model", learning_rate=1e-2):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        self.vocab_size = 0
        self.chars = []
        self.char_to_ix = {}
        self.ix_to_char = {}
        
        # GRU Weights
        # We process inputs (x) and hidden (h) together where possible
        self.Wz = None; self.Wr = None; self.Wh = None
        self.bz = None; self.br = None; self.bh = None
        self.Why = None; self.by = None

    def load_data(self, dataset_path):
        print(f"[Data] Reading {dataset_path}...")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_data = []
                for line in f:
                    try:
                        j = json.loads(line)
                        # Train on both Instruction and Response for better context
                        # But focus on Response for generation style
                        raw_data.append(j['response']) 
                    except: pass
                text = "\n".join(raw_data)
        except:
             # Fallback to plain text reading if not JSONL
             with open(dataset_path, 'r', encoding='utf-8') as f:
                 text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.chars = chars
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }
        print(f"[Data] Length: {len(text)} chars | Vocab: {self.vocab_size}")
        return text

    def init_model(self):
        # Xavier Initialization
        input_dim = self.vocab_size + self.hidden_size
        
        def xavier(r, c):
            return np.random.randn(r, c) * np.sqrt(2.0/(r+c))
            
        # Update Gate
        self.Wz = xavier(self.hidden_size, input_dim)
        self.bz = np.zeros((self.hidden_size, 1))
        
        # Reset Gate
        self.Wr = xavier(self.hidden_size, input_dim)
        self.br = np.zeros((self.hidden_size, 1))
        
        # New Memory (h_tilde)
        self.Wh = xavier(self.hidden_size, input_dim)
        self.bh = np.zeros((self.hidden_size, 1))
        
        # Output
        self.Why = xavier(self.vocab_size, self.hidden_size)
        self.by = np.zeros((self.vocab_size, 1))
        
        print("[Init] GRU Weights Initialized (GPU: " + str(USING_GPU) + ")")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, dataset_path, iterations=20000):
        data = self.load_data(dataset_path)
        self.init_model()
        
        # Adagrad Memory
        mWz, mWr, mWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
        mbz, mbr, mbh = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh)
        mWhy, mby = np.zeros_like(self.Why), np.zeros_like(self.by)
        
        smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length
        n, p = 0, 0
        hprev = np.zeros((self.hidden_size, 1))
        start = time.time()
        
        print(f"[Train] Starting {iterations} steps of GRU Deep Learning...")

        for i in range(iterations):
            # 1. Prepare Batch
            if p + self.seq_length + 1 >= len(data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            
            inputs = [self.char_to_ix[ch] for ch in data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]

            # 2. Forward Pass
            xs, hs, zs, rs, h_hats = {}, {}, {}, {}, {}
            hs[-1] = np.copy(hprev)
            loss = 0
            
            ps = {} # Probabilities
            
            for t in range(len(inputs)):
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][inputs[t]] = 1
                
                # GRU Logic
                combined = np.vstack((xs[t], hs[t-1]))
                
                zs[t] = self.sigmoid(np.dot(self.Wz, combined) + self.bz)
                rs[t] = self.sigmoid(np.dot(self.Wr, combined) + self.br)
                
                combined_r = np.vstack((xs[t], rs[t] * hs[t-1]))
                h_hats[t] = np.tanh(np.dot(self.Wh, combined_r) + self.bh)
                
                hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_hats[t]
                
                ys = np.dot(self.Why, hs[t]) + self.by
                ps[t] = np.exp(ys) / np.sum(np.exp(ys))
                loss += -np.log(ps[t][targets[t], 0])
            
            # 3. Backward Pass (BPTT - Simplified for brevity/stability)
            dWz, dWr, dWh = np.zeros_like(self.Wz), np.zeros_like(self.Wr), np.zeros_like(self.Wh)
            dWhy = np.zeros_like(self.Why)
            dbz, dbr, dbh = np.zeros_like(self.bz), np.zeros_like(self.br), np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])
            
            for t in reversed(range(len(inputs))):
                dy = np.copy(ps[t])
                dy[targets[t]] -= 1
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                
                dh = np.dot(self.Why.T, dy) + dh_next
                
                # Backprop through GRU cell (Approximate/Efficient version)
                # Calculating gradients for z, r, h_hat
                d_h_hat = dh * zs[t] * (1 - h_hats[t]**2)
                d_z = dh * (h_hats[t] - hs[t-1]) * zs[t] * (1 - zs[t])
                
                combined_r = np.vstack((xs[t], rs[t] * hs[t-1]))
                dWh += np.dot(d_h_hat, combined_r.T)
                dbh += d_h_hat
                
                combined = np.vstack((xs[t], hs[t-1]))
                dWz += np.dot(d_z, combined.T)
                dbz += d_z
                
                # Reset gate gradient (Simplification: assuming dominance of update gate for learning text)
                # Full BPTT for Reset gate is extremely expensive in Python loops
                # We update Reset gate based on prediction error flow through h_hat
                dr = np.dot(self.Wh[:, self.vocab_size:].T, d_h_hat) * hs[t-1] * rs[t] * (1 - rs[t])
                dWr += np.dot(dr, combined.T)
                dbr += dr
                
                # Next hidden state gradient approximation
                dh_next = dh * (1 - zs[t])
            
            # Clip
            for d in [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby]:
                np.clip(d, -5, 5, out=d)
            
            # Adagrad Apply
            for param, dparam, mem in zip([self.Wz, self.Wr, self.Wh, self.Why, self.bz, self.br, self.bh, self.by],
                                          [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby],
                                          [mWz, mWr, mWh, mWhy, mbz, mbr, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
            
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            p += self.seq_length
            n += 1
            hprev = hs[len(inputs)-1]
            
            if i % 100 == 0:
                print(f"Step {i}/{iterations} | Loss: {smooth_loss:.4f}")
        
        self.save_model()

    def save_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        def cpu(x):
            try: return real_numpy.array(x.get())
            except: return x
            
        real_numpy.savez(f"{self.output_dir}/weights.npz",
            Wz=cpu(self.Wz), Wr=cpu(self.Wr), Wh=cpu(self.Wh), Why=cpu(self.Why),
            bz=cpu(self.bz), br=cpu(self.br), bh=cpu(self.bh), by=cpu(self.by),
            chars=self.chars, architecture="GRU"
        )
        print("[Save] Saved Professional GRU Model.")

if __name__ == "__main__":
    trainer = GRUTrainer(hidden_size=256, seq_length=50)
    # Deep Learning: 50,000 steps to really understand English nuances
    trainer.train("data/train.jsonl", iterations=50000)
