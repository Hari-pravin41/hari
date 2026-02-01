import os
import sys
import numpy as np
import json
import re
from flask import Flask, request, jsonify
from pyngrok import ngrok
from ai_engine.chat_gpt import GPTChat

app = Flask(__name__)
ai_brain = None

def run_with_ngrok(token):
    # AUTH
    ngrok.set_auth_token(token)
    public_url = ngrok.connect(5000).public_url
    print(f"\n" + "="*50)
    print(f"🌍 SERVER ONLINE: {public_url}")
    print(f"="*50 + "\n")
    
    # Load Brain
    try:
        brain = GPTChat("models/gpt_model")
        print("✅ Brain Loaded Successfully.")
    except Exception as e:
        print(f"❌ Brain Error: {e}")
        brain = None
        
    global ai_brain
    ai_brain = brain
    
    app.run(port=5000)

@app.route('/analyze', methods=['POST'])
def analyze():
    msg = request.form.get('message', '')
    print(f"📩 input: {msg}")
    
    if ai_brain:
        try: 
            # PROMPT ENGINEERING
            # We force the AI into a "User vs AI" dialogue mode
            prompt = f"User: {msg}\nAI:"
            
            # Generate (Limit to 100 tokens to stop rambling)
            raw_res = ai_brain.generate(prompt, max_tokens=100, temperature=0.6)
            
            # CLEANUP
            # The AI might output "AI: Hello User: Hi". We want only "Hello".
            # 1. Remove the prompt itself if echoed
            res = raw_res.replace(prompt, "").strip()
            
            # 2. Stop at the next "User:" or "Q:" (Hallucination Cutoff)
            stop_markers = ["User:", "Q:", "AI:", "\n\n"]
            for marker in stop_markers:
                if marker in res:
                    res = res.split(marker)[0]
            
            # 3. Fallback if empty
            if not res: res = "I am thinking..."
            
        except Exception as e: 
            res = f"Error: {e}"
    else: 
        res = "Brain Offline (Training?)"
        
    return jsonify({"reply": res})

if __name__ == "__main__":
    print("Use the Colab Notebook to run this with GPU support.")
