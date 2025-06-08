#!/usr/bin/env python3
"""
Overnight Training Launcher
Choose your training intensity and duration for overnight runs.
"""

import sys
import os
import datetime

def show_training_options():
    """Display available training options."""
    print("🌙 Overnight Training Options")
    print("=" * 50)
    print()
    print("1. 🔥 MAXIMUM GPU Training (8 hours)")
    print("   • Pushes RTX 4090 to 95% utilization")
    print("   • ~150-200 turns/hour")
    print("   • 6 questions per document")
    print("   • Most intensive training")
    print()
    print("2. 🚀 Intensive Training (8 hours)")
    print("   • 85% GPU utilization")
    print("   • Multiple training cycles")
    print("   • 5 questions per document")
    print("   • Balanced performance/stability")
    print()
    print("3. 🔄 Continuous Training (8 hours)")
    print("   • Batch processing approach")
    print("   • Regular checkpointing")
    print("   • 3 questions per document")
    print("   • Most stable option")
    print()
    print("4. ⚡ Quick Test (30 minutes)")
    print("   • Test run before overnight")
    print("   • Verify everything works")
    print("   • 2 questions per document")
    print()
    print("5. 🛠️ Custom Training")
    print("   • Set your own parameters")
    print("   • Custom duration and intensity")
    print()

def launch_maximum_training():
    """Launch maximum GPU training."""
    print("🔥 Launching Maximum GPU Training...")
    os.system("python max_gpu_training.py")

def launch_intensive_training():
    """Launch intensive training."""
    print("🚀 Launching Intensive Training...")
    os.system("python overnight_gpu_training.py")

def launch_continuous_training():
    """Launch continuous training."""
    print("🔄 Launching Continuous Training...")
    
    # Create continuous training script call
    script = '''
import sys
sys.path.append(".")
from model import run_gpu_accelerated_training

print("🔄 Starting 8-hour Continuous GPU Training")
result = run_gpu_accelerated_training(
    num_turns=160,  # ~20 turns per hour for 8 hours
    batch_size=3,
    save_interval=10
)
print(f"✅ Training completed: {result}")
'''
    
    with open("temp_continuous.py", "w") as f:
        f.write(script)
    
    os.system("python temp_continuous.py")
    
    # Cleanup
    if os.path.exists("temp_continuous.py"):
        os.remove("temp_continuous.py")

def launch_quick_test():
    """Launch quick test."""
    print("⚡ Launching Quick Test...")
    
    script = '''
import sys
sys.path.append(".")
from model import run_gpu_accelerated_training

print("⚡ Starting 30-minute Test Run")
result = run_gpu_accelerated_training(
    num_turns=10,  # Quick test
    batch_size=2,
    save_interval=3
)
print(f"✅ Test completed: {result}")
'''
    
    with open("temp_test.py", "w") as f:
        f.write(script)
    
    os.system("python temp_test.py")
    
    # Cleanup
    if os.path.exists("temp_test.py"):
        os.remove("temp_test.py")

def launch_custom_training():
    """Launch custom training with user parameters."""
    print("🛠️ Custom Training Setup")
    print("-" * 30)
    
    try:
        hours = float(input("Training duration (hours): "))
        batch_size = int(input("Batch size (1-4): "))
        questions = int(input("Questions per document (2-8): "))
        
        estimated_turns = int(hours * 20 * batch_size)
        
        print(f"\n📊 Custom Configuration:")
        print(f"   ⏱️ Duration: {hours} hours")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   ❓ Questions per doc: {questions}")
        print(f"   🔄 Estimated turns: {estimated_turns}")
        
        confirm = input(f"\nProceed with custom training? (y/n): ")
        if confirm.lower() == 'y':
            script = f'''
import sys
sys.path.append(".")
from model import run_gpu_accelerated_training

print("🛠️ Starting Custom GPU Training")
print("   Duration: {hours} hours")
print("   Batch size: {batch_size}")
print("   Questions per doc: {questions}")

result = run_gpu_accelerated_training(
    num_turns={estimated_turns},
    batch_size={batch_size},
    save_interval=max(5, {estimated_turns}//20)
)
print(f"✅ Custom training completed: {{result}}")
'''
            
            with open("temp_custom.py", "w") as f:
                f.write(script)
            
            os.system("python temp_custom.py")
            
            # Cleanup
            if os.path.exists("temp_custom.py"):
                os.remove("temp_custom.py")
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")

def main():
    """Main launcher function."""
    print("🎯 AI Model Training Launcher")
    print(f"🕐 Current time: {datetime.datetime.now()}")
    print()
    
    while True:
        show_training_options()
        
        try:
            choice = input("Select training option (1-5, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                break
            
            elif choice == '1':
                confirm = input("⚠️ This will push your GPU to maximum limits. Continue? (y/n): ")
                if confirm.lower() == 'y':
                    launch_maximum_training()
                    break
            
            elif choice == '2':
                confirm = input("🚀 Start intensive 8-hour training? (y/n): ")
                if confirm.lower() == 'y':
                    launch_intensive_training()
                    break
            
            elif choice == '3':
                confirm = input("🔄 Start continuous 8-hour training? (y/n): ")
                if confirm.lower() == 'y':
                    launch_continuous_training()
                    break
            
            elif choice == '4':
                launch_quick_test()
                continue  # Don't exit after test
            
            elif choice == '5':
                launch_custom_training()
                continue  # Don't exit after custom
            
            else:
                print("❌ Invalid choice. Please select 1-5 or 'q'.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user.")
            break

if __name__ == "__main__":
    main()
