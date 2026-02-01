import os
import sys
import subprocess
import time
import threading

def run_server():
    print("[Bridge] Starting Local AI Server...")
    os.system(f'"{sys.executable}" ai_engine/server.py')

def start_tunnel():
    print("[Bridge] Establishing Secure Cloud Tunnel...")
    print("[Bridge] Installing LocalTunnel (if missing)...")
    
    # Check if npx is available
    # Check if npx is available
    npx_cmd = "npx.cmd" if os.name == 'nt' else "npx"
    try:
        subprocess.run([npx_cmd, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    except Exception as e:
        print(f"[Error] Node.js/NPX ({npx_cmd}) is not found. Error: {e}")
        return

    # Run LocalTunnel
    # We use 'lt' command via npx
    cmd = [npx_cmd, "localtunnel", "--port", "8000"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    
    print("[Bridge] Waiting for Tunnel URL...")
    while True:
        line = process.stdout.readline()
        if not line:
            break
        if "your url is" in line.lower():
            url = line.strip().split("is: ")[-1]
            print("\n" + "="*60)
            print(f"🚀  YOUR PROJECT IS LIVE ON THE CLOUD!")
            print("="*60)
            print(f"🌍  BACKEND URL: {url}")
            print("="*60)
            print("\nINSTRUCTIONS:")
            print("1. Deploy your Frontend to Vercel (free).")
            print(f"2. Add Environment Variable: AI_BACKEND_URL = {url}")
            print("3. Your Cloud App will now use YOUR Laptop's GPU!")
            print("\n(Press Ctrl+C to stop)")
        else:
            # Print other logs just in case
            if line.strip(): print(f"[LocalTunnel] {line.strip()}")

if __name__ == "__main__":
    t1 = threading.Thread(target=run_server)
    t1.start()
    
    # Give server a moment to start
    time.sleep(5)
    
    start_tunnel()
