import os
import time
import subprocess
import sys

def get_training_pid():
    # Use WMIC to find the python process running 'train_gru.py'
    try:
        cmd = 'wmic process where "name=\'python.exe\'" get commandline,processid'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        
        lines = output.strip().split('\n')
        for line in lines:
            if 'train_gru.py' in line:
                # Line format is usually: CommandLine   ProcessId
                # We need to extract the PID.
                # It's at the end of the line.
                parts = line.rsplit(None, 1) # Split from right to get PID
                if len(parts) > 0:
                    pid = parts[-1].strip()
                    if pid.isdigit():
                        return int(pid)
    except Exception as e:
        print(f"Error checking processes: {e}")
    return None

def trigger_sleep():
    print("Training finished. Going to sleep...")
    # Windows Sleep Command
    # This might hibernate if hibernation is on, which is usually fine/better for not losing data.
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

def monitor():
    print("Auto-Sleep Monitor Started.")
    print("Waiting for 'train_gru.py' to finish...")
    
    # 1. Find the process
    pid = get_training_pid()
    if not pid:
        print("Training process not found! Is it already finished?")
        # If it's already done (or I missed it), should I sleep now?
        # Maybe dangerous if I'm wrong. Safe bet: Don't sleep if I can't find it.
        # But maybe the user *just* started it and it's initializing.
        # Let's retry a few times.
        for _ in range(5):
            time.sleep(2)
            pid = get_training_pid()
            if pid: break
        
        if not pid:
            print("Could not find training process. Aborting auto-sleep to be safe.")
            return

    print(f"Tracking Training PID: {pid}")
    
    # 2. Monitor Loop
    while True:
        # Check if PID still exists
        try:
            # check_output logic again or simpler existence check
            cmd = f'tasklist /FI "PID eq {pid}"'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
            
            # If PID is not found, tasklist returns "No tasks are running..." or empty list in filter
            if str(pid) not in output:
                print("Process has exited.")
                break
        except:
             # If command fails, assume process might be gone or system issue
             break
             
        time.sleep(30) # Check every 30 seconds

    # 3. Sleep
    # Wait a moment to ensure file saves are flushed
    time.sleep(10)
    trigger_sleep()

if __name__ == "__main__":
    monitor()
