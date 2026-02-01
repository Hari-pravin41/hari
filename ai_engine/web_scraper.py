import requests
import json
import os
import re

class WebLearner:
    """
    Simulates 'Reading the Internet' or 'Reading Books'.
    Fetches text from URLs to expand the training dataset.
    """
    def __init__(self, dataset_path="data/train.jsonl"):
        self.dataset_path = dataset_path

    def clean_text(self, text):
        # Remove HTML tags manually (lightweight) if BS4 not installed
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def learn_from_url(self, url, source_name="Internet Source"):
        print(f"[WebLearner] Connecting to {url}...")
        try:
            # Simulate a browser
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                raw_text = self.clean_text(response.text)
                # Take a slice to avoid overwhelming the 200MB limit right now
                # In real life, we'd take it all.
                content = raw_text[:50000] 
                
                # Chunk it for training
                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                
                print(f"[WebLearner] Read {len(content)} characters from {source_name}.")
                print(f"[WebLearner] Digesting {len(chunks)} new facts...")
                
                # Append to dataset
                with open(self.dataset_path, 'a', encoding='utf-8') as f:
                    for chunk in chunks:
                        if len(chunk) > 50:
                            entry = {
                                "instruction": f"Read this excerpt from {source_name}:",
                                "response": chunk
                            }
                            f.write(json.dumps(entry) + "\n")
                            
                print("[WebLearner] Knowledge integrated successfully.")
                return True
            else:
                print(f"[WebLearner] Failed to fetch. Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[WebLearner] Error reading internet: {e}")
            return False

if __name__ == "__main__":
    # Test: Read a classic book snippet (Sherlock Holmes)
    learner = WebLearner()
    learner.learn_from_url("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", source_name="Book: Sherlock Holmes")
