import re
import json
import os
from collections import Counter

class SimpleTokenizer:
    def __init__(self, vocab_size=3000):
        self.vocab_size = vocab_size
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.unknown_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.vocab = []
        
    def clean_text(self, text):
        # Lowercase and split punctuation
        # "Hello, world!" -> "hello" "," "world" "!"
        text = text.lower()
        # Insert space around punctuation
        text = re.sub(r'([.,!?;:()\"\'\-])', r' \1 ', text)
        return text.split()

    def fit(self, text_data):
        print(f"[Tokenizer] Learning vocabulary from {len(text_data)} characters...")
        tokens = self.clean_text(text_data)
        word_counts = Counter(tokens)
        
        # Most common words
        common_words = word_counts.most_common(self.vocab_size - 2) # Reserve 2 for special tokens
        
        self.vocab = [self.pad_token, self.unknown_token] + [w for w, c in common_words]
        
        self.word_to_ix = {w: i for i, w in enumerate(self.vocab)}
        self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}
        
        print(f"[Tokenizer] Vocab built. Size: {len(self.vocab)}")
        print(f"[Tokenizer] Top 10 words: {self.vocab[:12]}")
        
    def encode(self, text):
        tokens = self.clean_text(text)
        return [self.word_to_ix.get(t, self.word_to_ix[self.unknown_token]) for t in tokens]
        
    def decode(self, indices):
        return " ".join([self.ix_to_word.get(ix, self.unknown_token) for ix in indices])
        
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "word_to_ix": self.word_to_ix,
                "ix_to_word": self.ix_to_word,
                "vocab_size": len(self.vocab)
            }, f)
            
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.word_to_ix = data["word_to_ix"]
            # JSON keys are always strings, convert indices back to int if needed (usually auto-handled for dict keys in json, wait no, keys are str)
            self.ix_to_word = {int(k): v for k, v in data["ix_to_word"].items()} 
            self.vocab_size = data["vocab_size"]
