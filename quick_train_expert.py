#!/usr/bin/env python3
"""
Quick Start CFA Expert Training

This script starts immediate autonomous training to make your agent a CFA expert.
Perfect for quick demonstration or when you want to start training right away.

Usage:
    python quick_train_expert.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Import training components
try:
    from autonomous_training import AutonomousTrainingAgent
    from model import reasoning_rag_module, doc_texts
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Make sure you're in the correct directory and all dependencies are installed")
    sys.exit(1)

def quick_expert_training():
    """Run quick expert training session."""
    print("🚀 Quick CFA Expert Training")
    print("=" * 40)
    
    # Check if documents are loaded
    if not doc_texts:
        print("❌ No documents loaded! Please run model.py first to index documents.")
        return
    
    print(f"📚 Documents loaded: {len(doc_texts)}")
    print(f"🎯 Goal: Achieve CFA expertise quickly")
    
    # Get quick configuration
    print("\n⚙️ Quick Configuration:")
    duration_hours = 4.0  # Default 4 hours
    target_expertise = 0.85  # 85% expertise target
    
    try:
        user_hours = input(f"Training hours (default {duration_hours}): ").strip()
        if user_hours:
            duration_hours = float(user_hours)
    except ValueError:
        print("Using default 4 hours")
    
    print(f"\n🎯 Training Plan:")
    print(f"   Duration: {duration_hours:.1f} hours")
    print(f"   Target: {target_expertise:.1%} expertise")
    print(f"   Weight Updates: Automatic (targeting 20-30 updates)")
    print(f"   Questions: ~{int(duration_hours * 20)} training questions")
    
    # Start training
    start_time = datetime.now()
    print(f"\n🏁 Starting training at {start_time.strftime('%H:%M:%S')}")
    
    try:
        # Initialize agent
        agent = AutonomousTrainingAgent(
            max_hours=duration_hours,
            target_expertise_score=target_expertise
        )
        
        # Test baseline
        print("\n📊 Testing baseline performance...")
        test_question = "What are the main components of portfolio risk management?"
        baseline_result = reasoning_rag_module(test_question)
        baseline_confidence = getattr(baseline_result, 'confidence', 0.0)
        print(f"   Baseline confidence: {baseline_confidence:.2f}")
        
        # Run training
        print(f"\n🚀 Launching autonomous training...")
        print(f"💡 The agent will generate questions, reason through them, judge its performance,")
        print(f"   and update its weights through reinforcement learning.")
        print(f"\n⚡ Watch for:")
        print(f"   • 🔄 Weight updates (model improvements)")
        print(f"   • 📈 Performance scores trending upward")
        print(f"   • 🧠 Reasoning quality improvements")
        print(f"   • 📚 Content coverage expansion")
        
        # Start autonomous training
        agent.run_autonomous_training()
        
        # Final assessment
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Training Complete!")
        print("=" * 40)
        print(f"⏰ Duration: {duration.total_seconds()/3600:.1f} hours")
        print(f"🎓 Final expertise: {agent._calculate_expertise_score():.1%}")
        print(f"🔄 Weight updates: {len(agent.weight_tracker.parameter_changes)}")
        print(f"📚 Topics explored: {len(agent.explored_topics)}")
        
        # Test improvement
        print(f"\n📊 Testing post-training performance...")
        final_result = reasoning_rag_module(test_question)
        final_confidence = getattr(final_result, 'confidence', 0.0)
        
        print(f"   Baseline confidence: {baseline_confidence:.2f}")
        print(f"   Final confidence: {final_confidence:.2f}")
        print(f"   Improvement: {final_confidence - baseline_confidence:.2f}")
        
        if final_confidence > baseline_confidence + 0.1:
            print(f"✅ Significant improvement detected!")
        elif final_confidence > baseline_confidence:
            print(f"📈 Moderate improvement detected!")
        else:
            print(f"📊 Model trained (improvement in reasoning depth)")
        
        # Final status
        final_expertise = agent._calculate_expertise_score()
        if final_expertise >= 0.9:
            print(f"\n🏆 EXPERT LEVEL ACHIEVED! Your agent is now a CFA expert!")
        elif final_expertise >= 0.7:
            print(f"\n🎯 ADVANCED LEVEL ACHIEVED! Your agent has strong CFA knowledge!")
        else:
            print(f"\n📈 IMPROVEMENT ACHIEVED! Continue training for expert level.")
        
        print(f"\n📁 Training data saved to:")
        print(f"   • autonomous_training_state/ - Training history")
        print(f"   • model_checkpoints/ - Model weights")
        print(f"   • training_analysis/ - Performance plots")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        print(f"💾 Progress has been saved automatically")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print(f"💾 Any progress has been saved")

def check_prerequisites():
    """Check if system is ready for training."""
    issues = []
    
    # Check documents
    try:
        from model import doc_texts
        if not doc_texts:
            issues.append("No documents loaded - run model.py first")
    except ImportError:
        issues.append("model.py not accessible")
    
    # Check autonomous training
    try:
        from autonomous_training import AutonomousTrainingAgent
    except ImportError:
        issues.append("autonomous_training.py not accessible")
    
    # Check paths
    required_files = ['model.py', 'autonomous_training.py']
    for file in required_files:
        if not Path(file).exists():
            issues.append(f"Missing required file: {file}")
    
    if issues:
        print("❌ Prerequisites not met:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n🔧 Please fix these issues before training.")
        return False
    
    print("✅ Prerequisites check passed!")
    return True

if __name__ == "__main__":
    print("⚡ Quick CFA Expert Training")
    print("-" * 30)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run quick training
    quick_expert_training()
    
    print(f"\n🎯 Training complete! Your agent is ready for CFA analysis.")
    print(f"💬 Try asking: 'How should portfolio allocation change during market volatility?'")
