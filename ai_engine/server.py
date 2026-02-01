import os
import sys
import numpy as np
import json
import re
from flask import Flask, request, jsonify
from pyngrok import ngrok
from ai_engine.chat_scratch import NumpyChat

app = Flask(__name__)

def run_with_ngrok(token):
    # AUTH
    ngrok.set_auth_token(token)
    public_url = ngrok.connect(5000).public_url
    print(f"\n" + "="*50)
    print(f"🌍 SERVER ONLINE: {public_url}")
    print(f"="*50 + "\n")
    
    # Load Brain
    try:
        # We assume the model was just trained and saved to models/token_model
        brain = NumpyChat("models/token_model")
        print("✅ Brain Loaded Successfully.")
    except Exception as e:
        print(f"❌ Brain Error: {e}")
        brain = None
        
    global ai_brain
    ai_brain = brain
    
    app.run(port=5000)

ai_brain = None

@app.route('/analyze', methods=['POST'])
def analyze():
    msg = request.form.get('message', '')
    print(f"📩 input: {msg}")
    
    if ai_brain:
        try: 
            # Temperature 0.7 for creativity
            res = ai_brain.generate(msg, temperature=0.7)
        except Exception as e: 
            res = f"Error: {e}"
    else: 
        res = "Brain Offline (Training?)"
        
    return jsonify({"reply": res})

if __name__ == "__main__":
    # For local testing
    print("Use the Colab Notebook to run this with GPU support.")
