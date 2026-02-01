import numpy as np

class GPTNumpy:
    """
    A 'High Level' Transformer Implementation from scratch using only Numpy.
    This mimics the architecture of GPT-2 roughly, implementing:
    - Multi-Head Self-Attention
    - Feed-Forward Networks
    - Layer Norm
    - Positional Embeddings
    
    Designed for educational/scratch purposes to run without heavy frameworks.
    """
    
    def __init__(self, vocab_size=100, d_model=64, n_layer=4, n_head=4, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.max_len = max_len
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Initialize Weights (Xavier/Glorot)
        self.params = {}
        
        # Embeddings
        self.params['wte'] = np.random.randn(vocab_size, d_model) * 0.02 # Token Embeddings
        self.params['wpe'] = np.random.randn(max_len, d_model) * 0.02    # Positional Embeddings
        
        # Transformer Blocks
        for i in range(n_layer):
            # Attention
            self.params[f'h{i}_c_attn'] = np.random.randn(d_model, 3*d_model) * 0.02 # Q,K,V projection combined
            self.params[f'h{i}_c_proj'] = np.random.randn(d_model, d_model) * 0.02   # Output projection
            self.params[f'h{i}_ln1_g'] = np.ones(d_model) # Layer norm gain
            self.params[f'h{i}_ln1_b'] = np.zeros(d_model) # Layer norm bias
            
            # MLP
            self.params[f'h{i}_c_fc'] = np.random.randn(d_model, 4*d_model) * 0.02
            self.params[f'h{i}_c_proj_mlp'] = np.random.randn(4*d_model, d_model) * 0.02
            self.params[f'h{i}_ln2_g'] = np.ones(d_model)
            self.params[f'h{i}_ln2_b'] = np.zeros(d_model)
            
        # Final Layer Norm
        self.params['ln_f_g'] = np.ones(d_model)
        self.params['ln_f_b'] = np.zeros(d_model)
        
        # Language Model Head (reuse embeddings often, but we'll separate for simplicity if needed, or reuse wte)
        # We reuse wte transposed usually
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def layer_norm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def attention(self, q, k, v, mask=None):
        # q, k, v: [seq_len, head_dim] (simplified for 1 batch)
        d_k = q.shape[-1]
        scores = np.matmul(q, k.T) / np.sqrt(d_k)
        
        if mask is not None:
             scores = np.where(mask == 0, -1e9, scores)
             
        attn = self.softmax(scores)
        return np.matmul(attn, v)

    def forward_block(self, x, i):
        # 1. Attention
        residual = x
        x_norm = self.layer_norm(x, self.params[f'h{i}_ln1_g'], self.params[f'h{i}_ln1_b'])
        
        # QKV Projection
        qkv = np.matmul(x_norm, self.params[f'h{i}_c_attn']) # [seq, 3*dim]
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # Split Heads (Simplification: treating as one big head for readability in this specific script, 
        # or we implement real multihead. Let's do Real Multihead).
        seq_len = x.shape[0]
        
        q = q.reshape(seq_len, self.n_head, self.head_dim).transpose(1, 0, 2) # [heads, seq, dim]
        k = k.reshape(seq_len, self.n_head, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_head, self.head_dim).transpose(1, 0, 2)
        
        # Self Attention with Causal Mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        
        out_heads = []
        for h in range(self.n_head):
            out_heads.append(self.attention(q[h], k[h], v[h], mask))
            
        # Concat heads
        attn_out = np.concatenate(out_heads, axis=-1) # [seq, dim]
        
        # Project Output
        attn_out = np.matmul(attn_out, self.params[f'h{i}_c_proj'])
        
        x = residual + attn_out
        
        # 2. MLP
        residual = x
        x_norm = self.layer_norm(x, self.params[f'h{i}_ln2_g'], self.params[f'h{i}_ln2_b'])
        
        # Feed FFN
        h_mlp = self.gelu(np.matmul(x_norm, self.params[f'h{i}_c_fc']))
        mlp_out = np.matmul(h_mlp, self.params[f'h{i}_c_proj_mlp'])
        
        x = residual + mlp_out
        return x

    def forward(self, idx):
        # idx: [seq_len] (list of integers)
        seq_len = len(idx)
        if seq_len > self.max_len:
            raise ValueError("Sequence too long")
            
        pos = np.arange(seq_len)
        
        # Token + Pos Embeddings
        tok_emb = self.params['wte'][idx] # [seq, dim]
        pos_emb = self.params['wpe'][pos]
        x = tok_emb + pos_emb
        
        # Blocks
        for i in range(self.n_layer):
            x = self.forward_block(x, i)
            
        # Final Norm
        x = self.layer_norm(x, self.params['ln_f_g'], self.params['ln_f_b'])
        
        # Head (using tie weights if standard, or just wte transpose)
        logits = np.matmul(x, self.params['wte'].T)
        return logits

    def generate(self, start_tokens, max_new_tokens=20):
        curr_ids = list(start_tokens)
        for _ in range(max_new_tokens):
            # Crop to max len
            cond_ids = curr_ids[-self.max_len:]
            
            # Forward
            logits = self.forward(cond_ids)
            next_token_logits = logits[-1] # Prediction for next token
            
            # Greedy or Sample
            next_id = np.argmax(next_token_logits)
            curr_ids.append(next_id)
            
        return curr_ids

    def load_weights(self, path):
        loaded = np.load(path)
        for k in self.params:
            if k in loaded:
                self.params[k] = loaded[k]
        print(f"Weights loaded from {path}")

    def save_weights(self, path):
        np.savez(path, **self.params)
