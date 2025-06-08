#!/usr/bin/env python3
"""
Quick training status checker - run anytime to see progress
"""

import os
import glob
import datetime
from pathlib import Path

def check_training_status():
    """Check current training progress"""
    print("🌙 OVERNIGHT TRAINING STATUS CHECK")
    print("=" * 50)
    
    # Check for recent checkpoints
    checkpoints = glob.glob("*checkpoint*.pkl")
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    if checkpoints:
        latest = checkpoints[0]
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(latest))
        time_ago = datetime.datetime.now() - mod_time
        
        print(f"📁 Latest checkpoint: {latest}")
        print(f"⏰ Last updated: {mod_time.strftime('%H:%M:%S')} ({time_ago.seconds//60}m ago)")
    else:
        print("⚠️ No checkpoints found yet")
    
    # Check log files
    log_files = glob.glob("overnight_logs/*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"📋 Latest log: {latest_log}")
        
        # Show last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                print(f"📊 Last 3 log entries:")
                for line in lines[-3:]:
                    print(f"   {line.strip()}")
        except:
            pass
    
    # Check GPU memory usage (if available)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                memory_used, memory_total, gpu_util = line.split(', ')
                print(f"🚀 GPU {i}: {memory_used}MB/{memory_total}MB ({float(memory_used)/float(memory_total)*100:.1f}%) | Utilization: {gpu_util}%")
    except:
        print("⚠️ Could not check GPU status")
    
    # Training time estimate
    start_time = datetime.datetime(2025, 6, 7, 23, 54, 52)  # When training started
    end_time = datetime.datetime(2025, 6, 8, 19, 54, 52)    # When training ends
    now = datetime.datetime.now()
    
    if now < end_time:
        remaining = end_time - now
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        print(f"⏳ Time remaining: {remaining.days * 24 + hours}h {minutes}m")
        
        elapsed = now - start_time
        total_duration = end_time - start_time
        progress = (elapsed.total_seconds() / total_duration.total_seconds()) * 100
        print(f"📈 Training progress: {progress:.1f}%")
    else:
        print("✅ Training should be completed!")

if __name__ == "__main__":
    check_training_status()
