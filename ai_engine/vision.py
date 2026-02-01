import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

class VisionEngine:
    """
    Professional Vision-Language Module for Image Diagnosis.
    Uses 'vikhyatk/moondream2' - a highly efficient <2GB VLM optimized for consumer GPUs.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-04-02" # Stable revision
        self.model = None
        self.tokenizer = None
        
    def analyze_image(self, image_bytes: bytes, prompt: str = "Describe this image in detail and diagnose any issues.") -> str:
        """
        Analyzes the image by calling the Persistent Vision Microservice (Port 8001).
        This is extremely fast as the model stays loaded.
        """
        import requests
        
        VISION_API_URL = "http://127.0.0.1:8001/analyze"
        
        try:
            print(f"[Vision Client] Sending request to Vision Microservice...")
            
            # Send Request
            response = requests.post(
                VISION_API_URL,
                data={"prompt": prompt},
                files={"image": ("image.png", image_bytes, "image/png")},
                timeout=120 # Wait up to 2 mins (just in case of interference, though usually sub-second)
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("answer", "No answer provided")
            else:
                return f"Vision Service Error {response.status_code}: {response.text}"

        except requests.exceptions.ConnectionError:
            return "Vision Service Offline! Please run 'python ai_engine/vision_server.py'."
        except Exception as e:
            return f"Vision Client Error: {e}"

    def unload(self):
        # No-op: The external service manages its own memory.
        pass
    
    def load_model(self):
        # No-op: handled by external service
        pass

if __name__ == "__main__":
    # Test stub
    eng = VisionEngine()
    print("Vision Engine Initialized.")
