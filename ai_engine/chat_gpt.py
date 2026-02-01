import torch
import os
import json
from ai_engine.gpt_torch import GPT
try:
    from ai_engine.tokenizer import SimpleTokenizer
except ImportError:
    from tokenizer import SimpleTokenizer

class GPTChat:
    def __init__(self, model_dir="models/gpt_model"):
        print(f"Loading GPT Brain from {model_dir}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Vocab
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(f"{model_dir}/vocab.json")
        
        # Load Model
        self.model = GPT(vocab_size=self.tokenizer.vocab_size)
        weights_path = f"{model_dir}/weights.pth"
        
        # Load Weights (handling CPU/GPU map)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("✅ GPT Brain Online.")

    def generate(self, prompt, max_tokens=50, temperature=0.7):
        # Encode
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generator
        with torch.no_grad():
            # We implemented generate() in the GPT class
            # But we might need to handle temperature manually if the raw class doesn't
            # (Our GPT class implementation above has basic generation)
            output_ids = self.model.generate(input_tensor, max_tokens)
            
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        return generated_text
