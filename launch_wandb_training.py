#!/usr/bin/env python3
"""
Quick launcher for WandB GPU Training
Launch and monitor your training with beautiful real-time graphs!
"""

import os
import sys
import subprocess
from datetime import datetime

def launch_wandb_training():
    """Launch WandB-monitored training"""
    print("🚀 Launching WandB GPU Training with Real-time Monitoring!")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("model.py"):
        print("❌ Error: model.py not found. Please run from S1 directory")
        return
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🔥 GPU Detected: {gpu_name}")
        else:
            print("⚠️  No GPU detected, training will use CPU")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    # Launch training
    print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 WandB dashboard will open automatically")
    print("📊 Real-time graphs: Loss, RL Scores, GPU Usage")
    print("💾 Auto-checkpointing every 25 turns")
    print("⏹️  Press Ctrl+C to stop training gracefully")
    print("=" * 60)
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, "wandb_gpu_training.py"], 
                              check=True, 
                              text=True)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("⏹️  Training stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    launch_wandb_training()
