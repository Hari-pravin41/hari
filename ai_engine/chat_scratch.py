import numpy as np
import sys
import os
import time
from tokenizer import SimpleTokenizer

class NumpyChat:
    def __init__(self, model_dir="./models/token_model"):
        print("[DEBUG] Loading Chat Engine v2.1 (Token Fix - AI Engine Path)...")
        self.model_path = f"{model_dir}/weights.npz"
        self.vocab_path = f"{model_dir}/vocab.json"
        
        if not os.path.exists(self.model_path):
            print("Token Model not found! Run training first.")
            sys.exit(1)
            
        print(f"[Chat] Loading Token-Level Brain from {model_dir}...")
        try:
            # Load Tokenizer
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.load(self.vocab_path)
            self.vocab_size = self.tokenizer.vocab_size
            
            # Load Weights
            data = np.load(self.model_path, allow_pickle=True)
            self.We = data['We']
            self.Wz = data['Wz']; self.Wr = data['Wr']; self.Wh = data['Wh']
            self.bz = data['bz']; self.br = data['br']; self.bh = data['bh']
            self.Why = data['Why']; self.by = data['by']
            
            self.hidden_size = self.bz.shape[0]
            print(f"[Chat] Brain Online (Architecture: Token-GRU, Vocab: {self.vocab_size}, Neurons: {self.hidden_size})")
            
        except Exception as e:
            print(f"[Chat] Error loading model: {e}")
            sys.exit(1)
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def generate(self, seed_text, length=50, temperature=0.7):
        """
        Token-Level AI Generation.
        Predicts whole words/concepts at a time.
        """
        # 1. Initialize Memory
        h = np.zeros((self.hidden_size, 1))
        
        # 2. Consume Content (Tokenize & Read)
        seed_tokens = self.tokenizer.encode(seed_text)
        
        if not seed_tokens:
             seed_tokens = [0] 

        for token_ix in seed_tokens:
            x = self.We[token_ix].reshape(-1, 1)
            
            # GRU Forward
            combined = np.vstack((x, h))
            z = self.sigmoid(np.dot(self.Wz, combined) + self.bz)
            r = self.sigmoid(np.dot(self.Wr, combined) + self.br)
            combined_r = np.vstack((x, r * h))
            candidate_h = np.tanh(np.dot(self.Wh, combined_r) + self.bh)
            h = (1 - z) * h + z * candidate_h

        # 3. Generate Response
        generated_tokens = []
        last_token = seed_tokens[-1]
        
        print("AI Thinking (Concept Association)...", end="", flush=True)
        time.sleep(0.5)
        
        for t in range(length):
            x = self.We[last_token].reshape(-1, 1)
            
            # GRU Step
            combined = np.vstack((x, h))
            z = self.sigmoid(np.dot(self.Wz, combined) + self.bz)
            r = self.sigmoid(np.dot(self.Wr, combined) + self.br)
            combined_r = np.vstack((x, r * h))
            candidate_h = np.tanh(np.dot(self.Wh, combined_r) + self.bh)
            h = (1 - z) * h + z * candidate_h
            
            # Output Logits
            y = np.dot(self.Why, h) + self.by
            
            # Softmax
            p = np.exp(y / temperature) / np.sum(np.exp(y / temperature))
            
            try:
                ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            except:
                ix = np.argmax(p)
                
            generated_tokens.append(ix)
            last_token = ix
            
            # Stop condition
            if ix == self.tokenizer.word_to_ix.get(".", 0) and len(generated_tokens) > 10:
                pass 
                
        text = self.tokenizer.decode(generated_tokens)
        return text

if __name__ == "__main__":
    chat = NumpyChat()
    print("\n[System] Token-Level Engine Online. Speaking in Words.")
    while True:
        try:
            q = input("\nYou: ")
            if q.lower() in ['exit', 'quit']: break
            print("AI: ", end="", flush=True)
            
            # Pass just the user query, normally we'd structure it "User: ... AI:"
            # But with simple tokenizer, keeping it clean is safer
            res = chat.generate(f"{q}", length=30) 
            
            print(res)
        except KeyboardInterrupt: break
