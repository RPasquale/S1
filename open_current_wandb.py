#!/usr/bin/env python3
"""
Direct WandB Dashboard Opener
Opens the current training run dashboard directly
"""

import webbrowser
import os
import glob
from datetime import datetime

def open_current_wandb_dashboard():
    """Open the current WandB training dashboard"""
    print("🌐 Opening WandB Dashboard for Current Training...")
    
    # Get the most recent wandb run
    wandb_runs = glob.glob("wandb/run-*")
    if wandb_runs:
        latest_run = max(wandb_runs, key=os.path.getmtime)
        run_id = os.path.basename(latest_run).split('-')[-1]
        
        # Construct WandB URL
        wandb_url = f"https://wandb.ai/robbie-pasquale/CFA-Dual-Training/runs/{run_id}"
        
        print(f"📊 Latest run: {latest_run}")
        print(f"🆔 Run ID: {run_id}")
        print(f"🌐 Dashboard URL: {wandb_url}")
        
        # Open in browser
        try:
            webbrowser.open(wandb_url)
            print("✅ Dashboard opened in browser!")
            print("\n🎯 You should now see real-time graphs showing:")
            print("   📉 NTP Loss (decreasing)")
            print("   📈 RL Score (increasing)")
            print("   🔥 GPU Utilization")
            print("   💾 GPU Memory Usage")
            print("   ⚡ Training Speed")
            print("   🖥️  System Metrics")
            
        except Exception as e:
            print(f"⚠️  Error opening browser: {e}")
            print(f"📋 Manual URL: {wandb_url}")
        
        return wandb_url
    else:
        print("❌ No WandB runs found")
        return None

if __name__ == "__main__":
    open_current_wandb_dashboard()
