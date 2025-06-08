#!/usr/bin/env python3
"""
🚀 OVERNIGHT TRAINING LAUNCHER
One-click script to start intensive overnight training
"""

import os
import sys
import subprocess
import datetime

def launch_overnight_training():
    """Launch overnight training with optimal settings"""
    print("🌙" + "="*60)
    print("🚀 OVERNIGHT TRAINING LAUNCHER - RTX 4090 OPTIMIZED")
    print("🌙" + "="*60)
    
    print("\n🎯 Training Options:")
    print("1. 🔥 MAXIMUM INTENSITY (10 hours, 8 questions/doc)")
    print("2. ⚡ HIGH INTENSITY (8 hours, 6 questions/doc)")
    print("3. 🔄 BALANCED (6 hours, 4 questions/doc)")
    print("4. 🛠️ CUSTOM SETTINGS")
    
    choice = input("\nSelect training intensity (1-4): ").strip()
    
    if choice == "1":
        # Maximum intensity overnight training
        print("\n🔥 MAXIMUM INTENSITY SELECTED")
        print("   - Duration: 10 hours")
        print("   - Questions per document: 8")
        print("   - GPU Memory usage: 85%")
        print("   - Estimated turns: ~200")
        print("   - Estimated questions: ~1,600")
        
        confirm = input("\n🚀 Start MAXIMUM intensity training? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            os.system("python ultimate_overnight_training.py max")
        
    elif choice == "2":
        # High intensity training
        print("\n⚡ HIGH INTENSITY SELECTED")
        print("   - Duration: 8 hours")
        print("   - Questions per document: 6")
        print("   - GPU Memory usage: 80%")
        print("   - Estimated turns: ~160")
        print("   - Estimated questions: ~960")
        
        confirm = input("\n🚀 Start HIGH intensity training? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            os.system("python ultimate_overnight_training.py high")
    
    elif choice == "3":
        # Balanced training
        print("\n🔄 BALANCED TRAINING SELECTED")
        print("   - Duration: 6 hours")
        print("   - Questions per document: 4")
        print("   - GPU Memory usage: 75%")
        print("   - Estimated turns: ~120")
        print("   - Estimated questions: ~480")
        
        confirm = input("\n🚀 Start BALANCED training? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            os.system("python ultimate_overnight_training.py 6")
    
    elif choice == "4":
        # Custom settings
        print("\n🛠️ CUSTOM SETTINGS")
        try:
            hours = float(input("Training duration (hours): "))
            
            print("\nIntensity levels:")
            print("- maximum: 8 questions/doc, 85% GPU")
            print("- high: 6 questions/doc, 80% GPU") 
            print("- moderate: 4 questions/doc, 75% GPU")
            
            intensity = input("Intensity level: ").strip().lower()
            
            if intensity not in ['maximum', 'high', 'moderate']:
                print("❌ Invalid intensity level")
                return
            
            print(f"\n🛠️ CUSTOM TRAINING: {hours}h, {intensity} intensity")
            confirm = input("🚀 Start custom training? (yes/no): ").strip().lower()
            
            if confirm in ['yes', 'y']:
                # Create custom command
                command = f"python -c \"from ultimate_overnight_training import ultimate_overnight_training; ultimate_overnight_training({hours}, '{intensity}')\""
                os.system(command)
                
        except ValueError:
            print("❌ Invalid input")
    
    else:
        print("❌ Invalid choice")

def launch_with_monitoring():
    """Launch training with simultaneous monitoring"""
    print("\n🔍 LAUNCH WITH MONITORING")
    print("This will start training and monitoring in parallel")
    
    confirm = input("🚀 Start training with monitoring? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        return
    
    # Start monitoring in background
    print("🔍 Starting monitoring...")
    monitor_process = subprocess.Popen([
        sys.executable, "training_monitor.py"
    ])
    
    # Give monitoring time to start
    import time
    time.sleep(2)
    
    # Start training
    print("🚀 Starting training...")
    launch_overnight_training()

def quick_start():
    """Quick start with recommended settings"""
    print("🚀 QUICK START - RECOMMENDED SETTINGS")
    print("   - Duration: 10 hours (full overnight)")
    print("   - Intensity: MAXIMUM (RTX 4090 optimized)")
    print("   - Questions per document: 8")
    print("   - GPU Memory: 85% utilization")
    print("   - Auto-checkpointing every 15 turns")
    
    current_time = datetime.datetime.now()
    end_time = current_time + datetime.timedelta(hours=10)
    
    print(f"\n⏰ Start time: {current_time.strftime('%H:%M:%S')}")
    print(f"⏰ End time: {end_time.strftime('%H:%M:%S')} (tomorrow)")
    
    confirm = input("\n🚀 Start QUICK overnight training? (yes/no): ").strip().lower()
    if confirm in ['yes', 'y']:
        print("🌙 Starting overnight training...")
        os.system("python ultimate_overnight_training.py quick")

def main():
    """Main launcher interface"""
    print("🌙 OVERNIGHT TRAINING LAUNCHER")
    print("="*50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_start()
        elif sys.argv[1] == "monitor":
            launch_with_monitoring()
        else:
            print("❌ Unknown argument. Use 'quick' or 'monitor'")
            return
    
    print("\n🎯 Launch Options:")
    print("1. 🚀 QUICK START (Recommended overnight settings)")
    print("2. 🔍 CUSTOM TRAINING (Choose your settings)")
    print("3. 📊 TRAINING + MONITORING (Parallel execution)")
    print("4. ❓ HELP (Show training recommendations)")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        quick_start()
    elif choice == "2":
        launch_overnight_training()
    elif choice == "3":
        launch_with_monitoring()
    elif choice == "4":
        show_help()
    else:
        print("❌ Invalid choice")

def show_help():
    """Show training recommendations and help"""
    print("\n" + "="*60)
    print("❓ TRAINING RECOMMENDATIONS")
    print("="*60)
    
    print("\n🚀 FOR RTX 4090 (24GB VRAM):")
    print("• MAXIMUM intensity: 8 questions/doc, 85% GPU memory")
    print("• Expected performance: ~20 turns/hour")
    print("• Recommended duration: 8-12 hours overnight")
    print("• Auto-checkpoints every 15 turns")
    
    print("\n📊 EXPECTED RESULTS:")
    print("• NTP Loss improvement: 15-25%")
    print("• RL Score improvement: 20-30%")
    print("• Total questions processed: 1,000-2,000")
    print("• All 14 training documents covered multiple times")
    
    print("\n⚠️  MONITORING RECOMMENDATIONS:")
    print("• Run monitoring script in parallel")
    print("• Check GPU temperature periodically")
    print("• Ensure adequate cooling")
    print("• Monitor for system stability")
    
    print("\n📁 FILES CREATED:")
    print("• Checkpoints: gpu_training_checkpoint_*.pkl")
    print("• Logs: overnight_logs/overnight_training_*.log")
    print("• Reports: ultimate_training_report_*.json")
    print("• Monitoring: training_monitor.json")

if __name__ == "__main__":
    main()
