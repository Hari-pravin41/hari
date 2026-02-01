import os
import json
import glob
from concurrent.futures import ThreadPoolExecutor

# Try importing PDF reader
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("[Importer] pypdf not installed. PDF files will be skipped.")

class DataIngestion:
    def __init__(self, output_file="data/train.jsonl"):
        self.output_file = output_file
        self.data_pairs = []
        self.CHUNK_SIZE = 500  # Characters
        self.MIN_CHUNK_LEN = 100

    def clean_text(self, text):
        """Removes excessive whitespace and non-printing characters."""
        return " ".join(text.split())

    def process_file(self, filepath):
        """Reads a file and converts it into training chunks."""
        content = ""
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext == '.txt' or ext == '.md' or ext == '.py' or ext == '.js' or ext == '.html':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            elif ext == '.pdf' and PDF_AVAILABLE:
                try:
                    reader = PdfReader(filepath)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                except Exception as e:
                    print(f"[Skip] Error reading PDF {filepath}: {e}")
                    return
            else:
                return # Skip unsupported
            
            # Post-processing
            content = self.clean_text(content)
            if len(content) < self.MIN_CHUNK_LEN:
                return

            # Create Slider Chunks
            # Strategy: "Context" -> "Continuation"
            # We split the file into chunks.
            # Instruction: [Text A]
            # Response:    [Text B] (The logical next part)
            
            step = self.CHUNK_SIZE
            for i in range(0, len(content) - step, step):
                chunk_inp = content[i : i+step]
                chunk_out = content[i+step : i+step*2]
                
                if len(chunk_out) < self.MIN_CHUNK_LEN:
                    break
                    
                self.data_pairs.append({
                    "instruction": f"Continue this text or code:\n{chunk_inp[-200:]}...", # Use last part of input as prompt
                    "response": chunk_out
                })
                
                # Also add a "Reading Comprehension" style pair
                # Instruction: "Read this context."
                # Response: [chunk_inp]
                # This helps it learn basic sentence structure
                self.data_pairs.append({
                    "instruction": "Read the following text:",
                    "response": chunk_inp
                })

            print(f"[Imported] {filepath} ({len(content)} chars)")
            
        except Exception as e:
            print(f"[Error] Failed to process {filepath}: {e}")

    def scan_directory(self, root_dir):
        """Recursively scans for readable files, SKIPPING TRASH."""
        print(f"[Scanner] Scanning {root_dir}...")
        files = []
        
        # Walk manually to control recursion
        for root, dirs, filenames in os.walk(root_dir):
            # 1. Block Trash Directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', '.git', '.vscode', '__pycache__', 'dist', 'build', '.next', '.venv', 'venv', 'site-packages'
            ]]
            
            for file in filenames:
                # 2. Block Trash Files
                if file in ['package-lock.json', 'yarn.lock', 'pnpm-lock.yaml']:
                    continue
                if file.endswith('.min.js') or file.endswith('.map'):
                    continue
                    
                # 3. Accept Logic
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.txt', '.md', '.py', '.json', '.pdf']:
                    files.append(os.path.join(root, file))
            
        print(f"[Scanner] Found {len(files)} clean, high-quality files.")
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.process_file, files)

    def save(self):
        """Appends new data to valid JSONL."""
        print(f"[Saver] Writing {len(self.data_pairs)} new training examples to {self.output_file}...")
        
        with open(self.output_file, 'a', encoding='utf-8') as f: # Append mode
            for item in self.data_pairs:
                f.write(json.dumps(item) + "\n")
        
        print(f"[Success] Dataset expanded! New size: {os.path.getsize(self.output_file) / 1024:.2f} KB")

# --- EXECUTION ---
if __name__ == "__main__":
    ingestor = DataIngestion()
    
    # 1. Ask user or Scan Default Paths
    # For now, we scan the PROJECT directory to self-learn the code
    # And maybe a Documents folder if accessible. 
    
    # Scan Project (Self-Learning)
    ingestor.scan_directory(".")
    
    # Scan Documents (Common Book location)
    docs_path = os.path.expanduser("~/Documents")
    if os.path.exists(docs_path):
        ingestor.scan_directory(docs_path)
    
    ingestor.save()
