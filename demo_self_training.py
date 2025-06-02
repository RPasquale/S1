#!/usr/bin/env python3
"""
Demonstration script for the Self-Training RAG System with Judge Model

This script demonstrates the complete self-training pipeline:
1. Question Generation from documents
2. Multi-step Reasoning
3. Judge Model scoring
4. Reinforcement Learning optimization with GRPO
5. Model checkpointing and evaluation

Usage:
    python demo_self_training.py
"""

import sys
import os
import time
from model import (
    self_training_system, 
    run_self_training_session,
    get_judge_insights,
    generate_training_questions,
    load_best_checkpoint
)

def demo_question_generation():
    """Demonstrate the adaptive question generation."""
    print("🎯 DEMO 1: Question Generation from Documents")
    print("=" * 50)
    
    questions = generate_training_questions(3)
    
    print(f"Generated {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. Question: {q['question']}")
        print(f"   Type: {q['type']}")
        print(f"   Difficulty: {q['difficulty']}")
        print(f"   Complexity: {q['complexity']}")
        print(f"   Expected Steps: {len(q['reasoning_steps'])} reasoning steps")

def demo_single_training_turn():
    """Demonstrate a single training turn."""
    print("\n🎯 DEMO 2: Single Training Turn")
    print("=" * 50)
    
    # Get a sample document
    if self_training_system.train_docs:
        doc_id, doc_content = self_training_system.train_docs[0]
        doc_sample = doc_content[:1500]  # First 1500 chars
        
        print(f"📄 Using document: {doc_id}")
        print(f"📝 Document preview: {doc_sample[:200]}...")
        
        # Execute one training turn
        print("\n🔄 Executing training turn...")
        turn_data = self_training_system.generate_training_turn(doc_sample, num_questions=2)
        
        # Display results
        print(f"\n📊 Turn Results:")
        print(f"- Questions generated: {len(turn_data['questions'])}")
        print(f"- Reasoning results: {len(turn_data['reasoning_results'])}")
        print(f"- Judge scores: {[f'{s:.3f}' for s in turn_data['judge_scores']]}")
        print(f"- Overall performance: {turn_data['overall_performance']:.3f}")
        
        # Show detailed results for first question
        if turn_data['reasoning_results']:
            result = turn_data['reasoning_results'][0]
            print(f"\n🔍 Detailed Result for First Question:")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['reasoning'].answer[:200]}...")
            print(f"Judge Score: {result['judge_score']:.3f}")
            print(f"Judge Feedback: {result['judge_feedback'][:200]}...")
    else:
        print("❌ No training documents available!")

def demo_judge_model():
    """Demonstrate the judge model scoring."""
    print("\n🎯 DEMO 3: Judge Model Insights")
    print("=" * 50)
    
    insights = get_judge_insights()
    print(insights)

def demo_self_training():
    """Demonstrate the full self-training loop."""
    print("\n🎯 DEMO 4: Self-Training Loop (3 turns)")
    print("=" * 50)
    
    print("🚀 Starting self-training with automatic checkpointing...")
    
    # Record start time
    start_time = time.time()
    
    # Run self-training
    result = run_self_training_session(3)
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{result}")
    print(f"⏱️ Training completed in {duration:.2f} seconds")
    
    # Show training status
    print(f"\n📈 Training Status:")
    print(self_training_system.get_training_status())

def demo_checkpoint_management():
    """Demonstrate checkpoint loading and management."""
    print("\n🎯 DEMO 5: Checkpoint Management")
    print("=" * 50)
    
    # Show best checkpoint
    best_result = load_best_checkpoint()
    print(f"🏆 {best_result}")
    
    # Show checkpoint history
    checkpoints = self_training_system.checkpoint_manager.checkpoint_history
    if checkpoints:
        print(f"\n💾 Checkpoint History ({len(checkpoints)} checkpoints):")
        for i, checkpoint in enumerate(checkpoints[-3:], 1):  # Show last 3
            print(f"  {i}. Step {checkpoint.step}: "
                  f"Train={checkpoint.train_score:.3f}, "
                  f"Val={checkpoint.val_score:.3f}, "
                  f"Test={checkpoint.test_score:.3f}")
    else:
        print("📝 No checkpoints available yet.")

def demo_adaptive_difficulty():
    """Demonstrate adaptive difficulty adjustment."""
    print("\n🎯 DEMO 6: Adaptive Difficulty System")
    print("=" * 50)
    
    generator = self_training_system.question_generator
    
    print("📊 Current difficulty distribution:")
    for level, prob in generator.difficulty_distribution.items():
        print(f"  {level}: {prob:.2f}")
    
    print("\n🎲 Simulating performance feedback...")
    
    # Simulate high performance (should increase difficulty)
    high_scores = [0.85, 0.90, 0.88, 0.92, 0.87]
    generator.adapt_difficulty(high_scores)
    
    print("After high performance (avg: 0.88):")
    for level, prob in generator.difficulty_distribution.items():
        print(f"  {level}: {prob:.2f}")
    
    # Simulate low performance (should decrease difficulty)
    low_scores = [0.35, 0.42, 0.38, 0.41, 0.39]
    generator.adapt_difficulty(low_scores)
    
    print("\nAfter low performance (avg: 0.39):")
    for level, prob in generator.difficulty_distribution.items():
        print(f"  {level}: {prob:.2f}")

def main():
    """Run all demonstrations."""
    print("🎉 Self-Training RAG System Demonstration")
    print("=" * 60)
    print("This demo showcases the complete self-training pipeline:")
    print("• Adaptive question generation")
    print("• Multi-step reasoning with judge scoring")
    print("• Reinforcement learning optimization")
    print("• Model checkpointing and evaluation")
    print("=" * 60)
    
    try:
        # Check if system is properly initialized
        if not hasattr(self_training_system, 'train_docs') or not self_training_system.train_docs:
            print("⚠️ Warning: No training documents found!")
            print("Please ensure documents are loaded before running demos.")
            return
        
        print(f"📚 System Status:")
        print(f"- Training docs: {len(self_training_system.train_docs)}")
        print(f"- Validation docs: {len(self_training_system.val_docs)}")
        print(f"- Test docs: {len(self_training_system.test_docs)}")
        
        # Run demonstrations
        demo_question_generation()
        demo_single_training_turn()
        demo_judge_model()
        demo_adaptive_difficulty()
        demo_self_training()
        demo_checkpoint_management()
        
        print("\n🎊 All demonstrations completed successfully!")
        print("\n💡 To interact with the system, run:")
        print("   python model.py")
        print("\n🔧 Available interactive commands:")
        print("   • self_train:5 - Run 5 training turns")
        print("   • judge_insights - View judge model insights")
        print("   • gen_questions:3 - Generate 3 training questions")
        print("   • training_status - Check current training status")
        print("   • best_checkpoint - Load best performing model")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("📝 Please check that all dependencies are installed and the system is properly configured.")

if __name__ == "__main__":
    main()
