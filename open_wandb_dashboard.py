#!/usr/bin/env python3
"""
Open WandB Dashboard
Quick script to open the WandB dashboard for monitoring
"""

import webbrowser
import json
import os
from pathlib import Path

def find_wandb_url():
    """Find the current WandB run URL"""
    # Look for recent checkpoint files
    checkpoint_files = list(Path('.').glob('wandb_checkpoint_*.pkl'))
    if not checkpoint_files:
        return None
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    
    try:
        import pickle
        with open(latest_checkpoint, 'rb') as f:
            data = pickle.load(f)
            return data.get('wandb_url')
    except:
        return None

def open_wandb_dashboard():
    """Open WandB dashboard"""
    print("🌐 Opening WandB Dashboard...")
    
    # Try to find existing URL
    url = find_wandb_url()
    
    if url:
        print(f"📊 Found existing run: {url}")
        webbrowser.open(url)
    else:
        # Open general WandB projects page
        webbrowser.open("https://wandb.ai/robbie-pasquale")
        print("📈 Opened WandB projects page")
        print("🔍 Look for 'Document-Dual-Training' project")

if __name__ == "__main__":
    open_wandb_dashboard()
