
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import sys
import os

# Define Service
app = FastAPI(title="RepairGPT Vision Microservice")

# Global Model State
model = None
tokenizer = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Vision Service] Starting on {DEVICE}. PID: {os.getpid()}")

def load_model():
    global model, tokenizer
    if model: return
    
    print("[Vision Service] Loading Moondream Model...")
    try:
        model_id = "vikhyatk/moondream2"
        model_id = "vikhyatk/moondream2"
        # Using latest revision as it is most compatible with current transformers
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Warmup
        model.eval()
        print(f"[Vision Service] Model Loaded & Ready (Latest).")
    except Exception as e:
        print(f"[Vision Service] CRITICAL LOAD ERROR: {e}")
        sys.exit(1)

@app.on_event("startup")
async def startup_event():
    # Load on startup for maximum speed
    load_model()

@app.post("/analyze")
async def analyze_image(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    global model, tokenizer
    try:
        print(f"[Vision Service] Analyzing request with prompt: '{prompt}'")
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        enc_image = model.encode_image(pil_image)
        answer = model.answer_question(enc_image, prompt, tokenizer)
        
        print(f"[Vision Service] Generated Answer: {answer}")
        
        return {"answer": answer, "status": "success"}
    except Exception as e:
        print(f"[Vision Service] Error: {e}")
        return {"answer": f"Vision Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    # Run on Port 8001 to avoid conflict with Main Server (8000)
    uvicorn.run(app, host="127.0.0.1", port=8001)
