import json
import os
import sys
import time
import numpy as real_numpy

# --- PROFESSIONAL GPU SETUP (CUDA 13 Compatibility Patch) ---
# 1. Register the local patch folder (contains renamed DLLs)
patch_dir = os.path.abspath("dll_patch")
if os.path.exists(patch_dir):
    try:
        os.add_dll_directory(patch_dir)
        os.environ['PATH'] = patch_dir + ";" + os.environ['PATH']
        print(f"[GPU Setup] Applied DLL Patch from: {patch_dir}")
    except Exception:
        pass

# 2. Register real CUDA 13 paths
cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
cuda_lib = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
for path in [cuda_root, cuda_lib]:
    if os.path.exists(path):
        try:
            os.add_dll_directory(path)
            os.environ['PATH'] = path + ";" + os.environ['PATH']
        except Exception:
            pass

try:
    import cupy as np
    print("[GPU Setup] Testing CUDA connection...")
    
    # 1. Basic Memory Test
    np.array([1])
    
    # 2. Kernel Check (NVRTC test)
    try:
        test_kernel = np.ElementwiseKernel('float32 x', 'float32 y', 'y = x * x', 'test')
        test_kernel(np.array([2.0], dtype=np.float32))
        print("[Init] 🚀 NVIDIA RTX GPU DETECTED & ACTIVE! (Training via CuPy)")
        USING_GPU = True
    except Exception as e:
        print(f"[Init] NVRTC Error (Compiler): {e}")
        print("       Attempting to continue with limited GPU support...")
        USING_GPU = True
        
except Exception as e:
    import numpy as np
    print(f"[Init] GPU Check Failed: {e}")
    print("[Init] Fallback to CPU.")
    USING_GPU = False

class NumpyTrainer:
    """
    A lightweight Character-Level RNN trained from scratch.
    Uses CuPy (GPU) if available, otherwise Numpy (CPU).
    """
    
    def __init__(self, hidden_size=256, seq_length=50, output_dir="./models/numpy_model"):
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 
        self.learning_rate = 1e-2
        self.output_dir = output_dir
        
        # Model parameters (Weights and Biases)
        self.Wxh = None # Input to Hidden
        self.Whh = None # Hidden to Hidden
        self.Why = None # Hidden to Output
        self.bh = None  # Hidden bias
        self.by = None  # Output bias
        
        self.chars = []
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.vocab_size = 0
        
    def load_data(self, dataset_path):
        print(f"[Data] Loading {dataset_path}...")
        text_data = ""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Combine instruction and response
                    text_data += f"User: {item['instruction']}\nAI: {item['response']}\n\n"
        except Exception as e:
            print(f"[Error] Failed to read data: {e}")
            return None
            
        print(f"[Data] Loaded {len(text_data)} characters.")
        
        # Build Vocabulary
        self.chars = sorted(list(set(text_data)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }
        
        print(f"[Data] Vocab Size: {self.vocab_size} unique characters.")
        return text_data

    def init_model(self):
        # Initialize parameters with small random values
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        backend = "GPU (CuPy)" if USING_GPU else "CPU (Numpy)"
        print(f"[Init] Model initialized on {backend}.")

    def lossFun(self, inputs, targets, hprev):
        """
        Runs forward and backward pass through the RNN
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # Forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t],0])
            
        # Backward pass (BPTT)
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
            
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def train(self, dataset_path, iterations=15000):
        data = self.load_data(dataset_path)
        if not data: return
        
        self.init_model()
        
        print(f"[Training] Teaching RepairGPT on {len(data)} characters...")
        print(f"[Training] Brain Size: {self.hidden_size} neurons. Steps: {iterations}")
        print("[Training] Strategy: Aggressive Memorization (Overfitting) with GPU Acceleration.")
        
        n, p = 0, 0
        hprev = np.zeros((self.hidden_size, 1))
        
        # Memory variables for Adagrad
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)

        os.makedirs(self.output_dir, exist_ok=True)
        
        start_time = time.time()
        
        for i in range(iterations):
            # Prepare inputs
            if p + self.seq_length + 1 >= len(data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            
            inputs = [self.char_to_ix[ch] for ch in data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]
            
            # Forward + Backward
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            
            # Adagrad Update
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
            
            p += self.seq_length
            n += 1
            
            if i % 500 == 0:
                print(f"Step {i}/{iterations} | Loss: {float(loss):.4f} | Learning...")
                
            if i % 2500 == 0 and i > 0: # Sample less frequently with more steps
                self.sample(hprev, data[p], 200) # Longer sample
                
        print("[Training] Training Complete. Saving Model...")
        print(f"[Training] Completed in {time.time() - start_time:.2f}s")
        self.save_model()
        
    def sample(self, h, seed_char, n):
        """
        Generate text from the model
        """
        x = np.zeros((self.vocab_size, 1))
        x[self.char_to_ix[seed_char]] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            # Sample from distribution
            p_val = p.ravel()
            if USING_GPU: p_val = np.asnumpy(p_val) # Move to CPU for random.choice if using cupy
            # Re-normalize for numerical stability
            p_val = p_val / real_numpy.sum(p_val)
            ix = real_numpy.random.choice(range(self.vocab_size), p=p_val)
            
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
            
        txt = ''.join(self.ix_to_char[ix] for ix in ixes)
        print(f"--- Sample Output ---\n...{txt}\n---------------------")
        
    def save_model(self):
        # Convert to CPU Numpy for portable saving
        def to_cpu(arr):
            try:
                return real_numpy.array(arr.get())
            except AttributeError:
                return arr
                
        real_numpy.savez(f"{self.output_dir}/weights.npz", 
                 Wxh=to_cpu(self.Wxh), Whh=to_cpu(self.Whh), Why=to_cpu(self.Why), 
                 bh=to_cpu(self.bh), by=to_cpu(self.by), 
                 chars=self.chars)
        print(f"[Save] Model saved to {self.output_dir}/weights.npz")

if __name__ == "__main__":
    trainer = NumpyTrainer()
    trainer.train("data/train.jsonl", iterations=15000)
