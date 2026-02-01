import re
from typing import List

class TextPreprocessor:
    """
    Handles text cleaning and tokenization preparation for the AI engine.
    """
    
    def __init__(self):
        # Placeholder for clearer initialization if needed (e.g. stop words)
        pass

    def clean_text(self, text: str) -> str:
        """
        Removes special characters, extra spaces, and normalizes text.
        """
        if not text:
            return ""
            
        # Lowercase
        text = text.lower()
        
        # Remove special characters (keep alphanumeric and basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def basic_tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer for preliminary processing.
        Real tokenization happens inside the Transformer model.
        """
        clean = self.clean_text(text)
        return clean.split()

# Example usage (Safe mode - no execution on import)
if __name__ == "__main__":
    processor = TextPreprocessor()
    sample = "Hello!!!   World -- This is AI."
    print(f"Original: {sample}")
    print(f"Cleaned: {processor.clean_text(sample)}")
