#!/usr/bin/env python3
"""
CFA Expert Training Launcher

This script provides a comprehensive menu to train your agent into a CFA expert
with explicit weight tracking and performance monitoring.

Usage:
    python train_cfa_expert.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_prerequisites():
    """Check if all required files and dependencies are available."""
    print("🔍 Checking Prerequisites...")
    required_files = [
        'model.py',
        'autonomous_training.py', 
        'run_long_training.py',
        'quick_train_expert.py',
        'monitor_training.py',
        'unlimited_training_wnb.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        return False
    
    # Check if documents are indexed
    try:
        sys.path.insert(0, '.')
        from model import doc_texts
        if not doc_texts:
            print("⚠️ No documents loaded. You'll need to run model.py first to index documents.")
            return False
        else:
            print(f"✅ {len(doc_texts)} documents loaded and ready")
    except ImportError as e:
        print(f"❌ Cannot import model: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Document check failed: {e}")
        return False
    
    print("✅ All prerequisites satisfied!")
    return True

def display_menu():
    """Display the main training menu."""
    print("\n" + "="*60)
    print("🎓 CFA EXPERT TRAINING CENTER")
    print("="*60)
    print("Transform your AI agent into a true CFA expert!")
    print()
    print("📚 Training Options:")
    print("  1. 🚀 Quick Expert Training (4 hours)")
    print("  2. 🏆 Ultimate Long Training (8-72 hours)")
    print("  3. 🌟 Unlimited WnB Training (Convergence-based)")
    print("  4. 🔍 Monitor Active Training")
    print("  5. 📊 View Training Status")
    print("  6. 💾 Load Best Checkpoint")
    print("  7. 🧠 Test Current Agent")
    print("  8. 📋 System Information")
    print("  9. ❌ Exit")
    print()

def quick_training():
    """Launch quick training session."""
    print("\n🚀 QUICK EXPERT TRAINING")
    print("-" * 40)
    print("This will train your agent for ~4 hours to achieve CFA expertise")
    print("Features:")
    print("  • 4-hour focused training session")
    print("  • Comprehensive weight tracking")
    print("  • CFA expertise development")
    print("  • Performance monitoring")
    print("  • Automatic checkpointing")
    
    confirm = input("\nLaunch quick training? (y/n): ").lower().strip()
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, 'quick_train_expert.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted")
    else:
        print("Training cancelled")

def ultimate_training():
    """Launch ultimate long training session."""
    print("\n🏆 ULTIMATE LONG TRAINING")
    print("-" * 40)
    print("This will run extended training to achieve maximum CFA expertise")
    print("Features:")
    print("  • Configurable duration (8-72 hours)")
    print("  • Comprehensive weight update tracking")
    print("  • Multi-day training capability")
    print("  • Advanced expertise assessment")
    print("  • Automatic checkpoint management")
    
    confirm = input("\nLaunch ultimate training? (y/n): ").lower().strip()
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, 'run_long_training.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted")
    else:
        print("Training cancelled")

def unlimited_wnb_training():
    """Launch unlimited WnB training with convergence-based stopping."""
    print("\n🌟 UNLIMITED WnB TRAINING")
    print("-" * 40)
    print("This will run unlimited duration training with Weights & Biases integration")
    print("Features:")
    print("  • Unlimited duration (stops on convergence)")
    print("  • Real-time WnB visualization and tracking")
    print("  • Comprehensive metrics logging")
    print("  • Custom dashboards and charts")
    print("  • Intelligent early stopping")
    print("  • Automatic artifact management")
    print("  • Advanced convergence detection")
    
    print("\n📊 WnB Features:")
    print("  • Live training metrics visualization")
    print("  • Loss and reward trend tracking")
    print("  • Weight update analysis")
    print("  • Performance comparison charts")
    print("  • Automatic model checkpointing")
    
    confirm = input("\nLaunch unlimited WnB training? (y/n): ").lower().strip()
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, 'unlimited_training_wnb.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ WnB Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted")
    else:
        print("Training cancelled")

def monitor_training():
    """Launch training monitor."""
    print("\n🔍 TRAINING MONITOR")
    print("-" * 40)
    print("Real-time monitoring of active training sessions")
    print("Shows:")
    print("  • Live weight updates")
    print("  • Performance trends")
    print("  • Expertise progression")
    print("  • Content exploration")
    
    try:
        subprocess.run([sys.executable, 'monitor_training.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Monitor failed: {e}")
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring stopped")

def view_status():
    """View current training status."""
    print("\n📊 TRAINING STATUS")
    print("-" * 40)
    
    # Check for training artifacts
    checkpoints_dir = Path('model_checkpoints')
    training_state_dir = Path('autonomous_training_state') 
    ultimate_sessions = list(Path('.').glob('ultimate_training_*'))
    
    print(f"🔍 Training Artifacts Found:")
    
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob('*.json'))
        print(f"  • Model Checkpoints: {len(checkpoints)} files")
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"    └ Latest: {latest.name}")
    else:
        print("  • Model Checkpoints: None")
    
    if training_state_dir.exists():
        state_files = list(training_state_dir.glob('*.json'))
        print(f"  • Training States: {len(state_files)} files")
    else:
        print("  • Training States: None")
    
    if ultimate_sessions:
        print(f"  • Ultimate Sessions: {len(ultimate_sessions)} directories")
        for session in ultimate_sessions[:3]:  # Show latest 3
            print(f"    └ {session.name}")
    else:
        print("  • Ultimate Sessions: None")
    
    # Check running processes
    print(f"\n🔄 System Status:")
    print(f"  • Python Path: {sys.executable}")
    print(f"  • Working Directory: {Path.cwd()}")
    print(f"  • Available Memory: Checking...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"    └ {memory.available // (1024**3)} GB available")
    except ImportError:
        print("    └ Install psutil for memory info")

def load_best_checkpoint():
    """Load the best checkpoint."""
    print("\n💾 LOAD BEST CHECKPOINT")
    print("-" * 40)
    
    try:
        from model import load_best_checkpoint as load_best
        result = load_best()
        print(result)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")

def test_agent():
    """Test the current agent with sample questions."""
    print("\n🧠 AGENT TESTING")
    print("-" * 40)
    
    cfa_questions = [
        "What are the key principles of portfolio diversification?",
        "How do interest rate changes affect bond pricing?",
        "What factors should be considered in asset allocation?",
        "Explain the difference between systematic and unsystematic risk.",
        "How does market volatility impact investment strategies?"
    ]
    
    print("Testing agent with CFA questions...")
    
    try:
        from model import reasoning_rag_module
        
        for i, question in enumerate(cfa_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            
            start_time = time.time()
            result = reasoning_rag_module(question)
            response_time = time.time() - start_time
            
            answer = result.answer if hasattr(result, 'answer') else str(result)
            confidence = getattr(result, 'confidence', 0.0)
            
            print(f"⏱️ Response Time: {response_time:.2f}s")
            print(f"🎯 Confidence: {confidence:.2f}")
            print(f"💬 Answer: {answer[:150]}...")
            
            if i < len(cfa_questions):
                input("Press Enter for next question...")
                
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def system_info():
    """Display system information."""
    print("\n📋 SYSTEM INFORMATION")
    print("-" * 40)
    
    try:
        from model import doc_texts, embeddings_model
        
        ready_items = []
        
        if doc_texts:
            ready_items.append(f"✅ Documents: {len(doc_texts)} loaded")
        else:
            ready_items.append("❌ Documents: Not loaded")
        
        if embeddings_model:
            ready_items.append("✅ Embeddings Model: Ready")
        else:
            ready_items.append("❌ Embeddings Model: Not loaded")
        
        if Path('model_checkpoints').exists():
            checkpoints = list(Path('model_checkpoints').glob('*.json'))
            ready_items.append(f"✅ Checkpoints: {len(checkpoints)} available")
        else:
            ready_items.append("❌ Checkpoints: None found")
        
        if Path('autonomous_training.py').exists():
            ready_items.append("✅ Training System: Available")
        else:
            ready_items.append("❌ Training System: Missing")
            
        if Path('unlimited_training_wnb.py').exists():
            ready_items.append("✅ WnB Training: Available")
        else:
            ready_items.append("❌ WnB Training: Missing")
        
        print("🔍 System Components:")
        for item in ready_items:
            print(f"  {item}")
        
        if len([item for item in ready_items if item.startswith("✅")]) >= 3:
            print(f"\n🚀 System is ready for CFA expert training!")
        else:
            print(f"\n⚠️ Some components missing - check prerequisites")
            
    except Exception as e:
        print(f"❌ System check failed: {e}")

def main():
    """Main menu loop."""
    print("🤖 CFA Expert Training System")
    print("🎯 Transform Your AI Into a CFA Expert")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix issues before proceeding.")
        return
    
    while True:
        display_menu()
        
        try:
            choice = input("Select option (1-9): ").strip()
            
            if choice == '1':
                quick_training()
            elif choice == '2':
                ultimate_training()
            elif choice == '3':
                unlimited_wnb_training()
            elif choice == '4':
                monitor_training()
            elif choice == '5':
                view_status()
            elif choice == '6':
                load_best_checkpoint()
            elif choice == '7':
                test_agent()
            elif choice == '8':
                system_info()
            elif choice == '9':
                print("\n👋 Goodbye! Your CFA expert awaits training.")
                break
            else:
                print("❌ Invalid option. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Pause before showing menu again
        if choice != '9':
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
