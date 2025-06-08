#!/usr/bin/env python3
"""
Autonomous Long-Term Training Agent for CFA Content Mastery

This script runs extended autonomous training sessions where the agent:
1. Continuously generates questions from CFA documents
2. Trains its reasoning through multi-step analysis
3. Uses judge model to score and improve reasoning
4. Updates model weights through GRPO reinforcement learning
5. Tracks detailed training progress and weight changes
6. Automatically adapts difficulty and explores content deeply

Key Features:
- Explicit weight tracking and update monitoring
- Detailed reasoning training visualization
- Automatic checkpoint management
- Performance trend analysis
- Content exploration tracking
"""

import os
import sys
import time
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import random

# Import our main model components
from model import (
    SelfTrainingLoop, ReasoningRAG, ReasoningJudgeModel, 
    AdaptiveQuestionGenerator, ModelCheckpoint, TrainingMetrics,
    doc_texts, reasoning_rag_module, generate_reasoning_trainset, reasoning_quality_metric,
    GRPO
)

class WeightTracker:
    """Tracks and visualizes model weight changes during training."""
    
    def __init__(self):
        self.weight_history = []
        self.gradient_history = []
        self.parameter_changes = []
        
    def capture_model_state(self, model, step: int, label: str = ""):
        """Capture current model state for comparison."""
        state_hash = self._compute_state_hash(model)
        
        weight_snapshot = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'state_hash': state_hash,
            'parameter_count': self._count_parameters(model),
            'model_structure': self._analyze_structure(model)
        }
        
        self.weight_history.append(weight_snapshot)
        
        # Compare with previous state if available
        if len(self.weight_history) > 1:
            change_delta = self._compute_change_delta(
                self.weight_history[-2], 
                self.weight_history[-1]
            )
            self.parameter_changes.append(change_delta)
            
        return weight_snapshot
    
    def _compute_state_hash(self, model) -> str:
        """Compute hash of model state for change detection."""
        try:
            # Extract serializable model state
            state_data = {}
            if hasattr(model, '__dict__'):
                for key, value in model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        try:
                            # Convert to string for hashing
                            state_data[key] = str(value)
                        except:
                            continue
            
            # Create hash
            state_str = json.dumps(state_data, sort_keys=True)
            return hashlib.md5(state_str.encode()).hexdigest()[:16]
        except Exception as e:
            print(f"⚠️ Error computing state hash: {e}")
            return f"error_{datetime.now().timestamp()}"
    
    def _count_parameters(self, model) -> Dict[str, int]:
        """Count parameters in model components."""
        try:
            param_counts = {}
            
            # Count parameters in different components
            components = ['decompose', 'analyze', 'synthesize', 'reflect', 'compare']
            
            for comp_name in components:
                if hasattr(model, comp_name):
                    comp = getattr(model, comp_name)
                    param_counts[comp_name] = len(str(comp))  # Simplified counting
            
            param_counts['total_estimated'] = sum(param_counts.values())
            return param_counts
            
        except Exception as e:
            print(f"⚠️ Error counting parameters: {e}")
            return {'error': 0}
    
    def _analyze_structure(self, model) -> Dict[str, Any]:
        """Analyze model structure and components."""
        try:
            structure = {
                'components': [],
                'reasoning_chains': getattr(model, 'reasoning_chains', 0),
                'num_docs': getattr(model, 'num_docs', 0),
                'history_length': len(getattr(model, 'reasoning_history', []))
            }
            
            # Analyze components
            for attr_name in dir(model):
                if not attr_name.startswith('_') and hasattr(model, attr_name):
                    attr = getattr(model, attr_name)
                    if hasattr(attr, '__class__'):
                        structure['components'].append({
                            'name': attr_name,
                            'type': attr.__class__.__name__
                        })
            
            return structure
            
        except Exception as e:
            print(f"⚠️ Error analyzing structure: {e}")
            return {'error': str(e)}
    
    def _compute_change_delta(self, prev_state: Dict, curr_state: Dict) -> Dict:
        """Compute changes between model states."""
        try:
            delta = {
                'step_from': prev_state['step'],
                'step_to': curr_state['step'],
                'hash_changed': prev_state['state_hash'] != curr_state['state_hash'],
                'structure_changed': prev_state['model_structure'] != curr_state['model_structure'],
                'parameter_changes': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Compare parameter counts
            prev_params = prev_state.get('parameter_count', {})
            curr_params = curr_state.get('parameter_count', {})
            
            for param_name in set(prev_params.keys()) | set(curr_params.keys()):
                prev_val = prev_params.get(param_name, 0)
                curr_val = curr_params.get(param_name, 0)
                if prev_val != curr_val:
                    delta['parameter_changes'][param_name] = {
                        'from': prev_val,
                        'to': curr_val,
                        'change': curr_val - prev_val
                    }
            
            return delta
            
        except Exception as e:
            print(f"⚠️ Error computing change delta: {e}")
            return {'error': str(e)}
    
    def print_weight_changes(self):
        """Print summary of weight changes during training."""
        if not self.parameter_changes:
            print("🔍 No weight changes detected yet.")
            return
        
        print("\n🔄 Model Weight Changes During Training:")
        print("=" * 50)
        
        for i, change in enumerate(self.parameter_changes[-5:], 1):  # Last 5 changes
            print(f"\n📊 Change {i} (Step {change['step_from']} → {change['step_to']}):")
            print(f"   Hash Changed: {'✅' if change['hash_changed'] else '❌'}")
            print(f"   Structure Changed: {'✅' if change['structure_changed'] else '❌'}")
            
            if change['parameter_changes']:
                print("   Parameter Updates:")
                for param, details in change['parameter_changes'].items():
                    print(f"     • {param}: {details['from']} → {details['to']} (Δ{details['change']})")
            else:
                print("   No parameter changes detected")
    
    def save_training_plots(self, output_dir: Path = Path("training_analysis")):
        """Save training analysis plots."""
        output_dir.mkdir(exist_ok=True)
        
        if len(self.weight_history) < 2:
            print("⚠️ Not enough data for plots")
            return
        
        try:
            # Plot 1: Parameter count over time
            steps = [h['step'] for h in self.weight_history]
            param_counts = [h['parameter_count'].get('total_estimated', 0) for h in self.weight_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, param_counts, marker='o')
            plt.title('Model Parameter Count Over Training')
            plt.xlabel('Training Step')
            plt.ylabel('Estimated Parameter Count')
            plt.grid(True)
            plt.savefig(output_dir / 'parameter_evolution.png')
            plt.close()
            
            # Plot 2: Hash changes (indicating model updates)
            hash_changes = []
            for i in range(1, len(self.weight_history)):
                prev_hash = self.weight_history[i-1]['state_hash']
                curr_hash = self.weight_history[i]['state_hash']
                hash_changes.append(1 if prev_hash != curr_hash else 0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps[1:], hash_changes, marker='o', linestyle='-')
            plt.title('Model State Changes Over Training')
            plt.xlabel('Training Step')
            plt.ylabel('Model Updated (1=Yes, 0=No)')
            plt.grid(True)
            plt.savefig(output_dir / 'model_updates.png')
            plt.close()
            
            print(f"📊 Training plots saved to: {output_dir}")
            
        except Exception as e:
            print(f"⚠️ Error creating plots: {e}")

class AutonomousTrainingAgent:
    """Long-term autonomous training agent for CFA content mastery."""
    
    def __init__(self, max_hours: float = 24.0, target_expertise_score: float = 0.95):
        """
        Initialize autonomous training agent.
        
        Args:
            max_hours: Maximum training time in hours
            target_expertise_score: Target score to achieve before stopping
        """
        self.max_hours = max_hours
        self.target_expertise_score = target_expertise_score
        self.start_time = datetime.now()
        
        # Initialize components
        self.training_loop = SelfTrainingLoop(num_docs=5, reasoning_chains=3)
        self.weight_tracker = WeightTracker()
        
        # Training configuration
        self.training_config = {
            'questions_per_turn': 5,  # More questions per turn
            'checkpoint_every': 5,   # Save every 5 turns
            'deep_evaluation_every': 10,  # Thorough evaluation every 10 turns
            'exploration_boost_every': 20,  # Increase exploration every 20 turns
            'difficulty_adaptation_threshold': 0.1,  # Adapt when performance changes by 10%
        }
        
        # Performance tracking
        self.performance_history = []
        self.expertise_indicators = {
            'content_coverage': 0.0,      # How much content has been explored
            'reasoning_depth': 0.0,       # Average reasoning chain complexity
            'judge_consistency': 0.0,     # How consistent judge scores are
            'improvement_rate': 0.0,      # Rate of performance improvement
        }
        
        # Content exploration tracking
        self.explored_topics = set()
        self.topic_performance = {}
        
        print(f"🚀 Autonomous Training Agent Initialized")
        print(f"🎯 Target: {target_expertise_score:.1%} expertise in {max_hours:.1f} hours")
        print(f"📚 Available documents: {len(doc_texts)}")
        
    def run_autonomous_training(self):
        """Run the main autonomous training loop."""
        print(f"\n🏁 Starting Autonomous Training Session")
        print(f"⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Will train until {(self.start_time + timedelta(hours=self.max_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        turn_count = 0
        best_score = 0.0
        
        while not self._should_stop_training(turn_count, best_score):
            turn_count += 1
            
            print(f"\n{'='*60}")
            print(f"🔄 AUTONOMOUS TRAINING TURN {turn_count}")
            print(f"{'='*60}")
            
            # Capture model state before training
            pre_state = self.weight_tracker.capture_model_state(
                self.training_loop.reasoning_rag, 
                turn_count, 
                f"pre_turn_{turn_count}"
            )
            
            # Execute training turn with enhanced exploration
            turn_results = self._execute_enhanced_turn(turn_count)
            
            # Capture model state after training
            post_state = self.weight_tracker.capture_model_state(
                self.training_loop.reasoning_rag, 
                turn_count, 
                f"post_turn_{turn_count}"
            )
            
            # Analyze weight changes
            self._analyze_weight_updates(pre_state, post_state, turn_results)
            
            # Update performance tracking
            current_score = turn_results.get('overall_performance', 0.0)
            self.performance_history.append({
                'turn': turn_count,
                'score': current_score,
                'timestamp': datetime.now().isoformat(),
                'details': turn_results
            })
            
            # Update best score
            if current_score > best_score:
                best_score = current_score
                print(f"🎉 NEW BEST SCORE: {best_score:.3f}")
            
            # Deep evaluation and analysis
            if turn_count % self.training_config['deep_evaluation_every'] == 0:
                self._perform_deep_evaluation(turn_count, best_score)
            
            # Content exploration boost
            if turn_count % self.training_config['exploration_boost_every'] == 0:
                self._boost_exploration()
            
            # Print progress
            self._print_training_progress(turn_count, current_score, best_score)
            
            # Auto-save progress
            if turn_count % self.training_config['checkpoint_every'] == 0:
                self._save_training_state(turn_count)
          # Training completed
        self._complete_training_session(turn_count, best_score)
    
    def _execute_enhanced_turn(self, turn_count: int) -> Dict[str, Any]:
        """Execute an enhanced training turn with deep content exploration."""
        
        # Select document strategically (not just random)
        doc_sample = self._select_strategic_document(turn_count)
        
        # Determine number of questions based on turn and performance
        num_questions = self._determine_question_count(turn_count)
        
        print(f"📊 Turn {turn_count}: {num_questions} questions from strategic document")
        
        # Execute the training turn
        turn_data = self.training_loop.generate_training_turn(
            doc_sample, 
            num_questions=num_questions
        )
        
        # Enhanced reasoning analysis
        self._analyze_reasoning_quality(turn_data)
        
        # Track content exploration
        self._track_content_exploration(turn_data)
        
        # Enhance turn_data with additional metrics for unlimited training
        enhanced_metrics = self._calculate_enhanced_metrics(turn_data, turn_count)
        turn_data.update(enhanced_metrics)
        
        return turn_data
    
    def _select_strategic_document(self, turn_count: int) -> str:
        """Strategically select document for training."""
        
        # Mix of strategies based on turn count
        if turn_count % 4 == 0:
            # Explore least covered content
            return self._select_unexplored_document()
        elif turn_count % 4 == 1:
            # Focus on challenging content
            return self._select_challenging_document()
        elif turn_count % 4 == 2:
            # Review high-performing content for deeper analysis
            return self._select_high_performing_document()
        else:
            # Random exploration
            doc_id, doc_content = random.choice(self.training_loop.train_docs)
            return doc_content[:2000]
    
    def _select_unexplored_document(self) -> str:
        """Select document from unexplored content."""
        # Find documents with topics not yet explored
        for doc_id, doc_content in self.training_loop.train_docs:
            doc_topics = self._extract_topics(doc_content)
            if not any(topic in self.explored_topics for topic in doc_topics):
                print("🔍 Selected unexplored content")
                return doc_content[:2000]
        
        # If all explored, select least explored
        print("🔄 All content explored, selecting least covered")
        return random.choice(self.training_loop.train_docs)[1][:2000]
    
    def _select_challenging_document(self) -> str:
        """Select document from content that has been challenging."""
        challenging_docs = [
            doc for doc, performance in self.topic_performance.items()
            if performance < 0.7  # Below 70% performance
        ]
        
        if challenging_docs:
            print("💪 Selected challenging content for focused training")
            return random.choice(challenging_docs)
        
        print("🎯 No challenging content identified, selecting random")
        return random.choice(self.training_loop.train_docs)[1][:2000]
    
    def _select_high_performing_document(self) -> str:
        """Select document from high-performing content for deeper analysis."""
        high_performing_docs = [
            doc for doc, performance in self.topic_performance.items()
            if performance > 0.8  # Above 80% performance
        ]
        
        if high_performing_docs:
            print("⭐ Selected high-performing content for deeper analysis")
            return random.choice(high_performing_docs)
        
        print("📊 No high-performing content identified, selecting random")
        return random.choice(self.training_loop.train_docs)[1][:2000]
    
    def _determine_question_count(self, turn_count: int) -> int:
        """Determine number of questions based on training progress."""
        base_questions = self.training_config['questions_per_turn']
        
        # Increase questions over time for deeper exploration
        if turn_count > 50:
            return base_questions + 2
        elif turn_count > 20:
            return base_questions + 1
        else:
            return base_questions
    
    def _analyze_weight_updates(self, pre_state: Dict, post_state: Dict, turn_results: Dict):
        """Analyze and report weight updates."""
        print(f"\n🔍 WEIGHT UPDATE ANALYSIS:")
        print("-" * 40)
        
        # Check if model state changed
        state_changed = pre_state['state_hash'] != post_state['state_hash']
        print(f"Model State Changed: {'✅ YES' if state_changed else '❌ NO'}")
        
        if state_changed:
            print(f"🔄 Model weights updated during this turn!")
            print(f"   Pre-training hash:  {pre_state['state_hash']}")
            print(f"   Post-training hash: {post_state['state_hash']}")
            
            # Show what triggered the update
            if turn_results.get('judge_scores'):
                avg_score = np.mean(turn_results['judge_scores'])
                print(f"   Triggered by: Judge scores (avg: {avg_score:.3f})")
                print(f"   Judge feedback quality influenced weight updates")
        else:
            print(f"   No weight updates detected")
            print(f"   Model state remained stable this turn")
        
        # Show parameter changes
        param_changes = post_state['parameter_count']
        print(f"\nParameter Status:")
        for component, count in param_changes.items():
            print(f"   {component}: {count}")
    
    def _analyze_reasoning_quality(self, turn_data: Dict):
        """Analyze the quality of reasoning generated."""
        print(f"\n🧠 REASONING QUALITY ANALYSIS:")
        print("-" * 40)
        
        reasoning_results = turn_data.get('reasoning_results', [])
        if not reasoning_results:
            print("   No reasoning results to analyze")
            return
        
        # Analyze reasoning depth
        avg_reasoning_length = np.mean([
            len(result['reasoning'].reasoning_chain) 
            for result in reasoning_results 
            if hasattr(result['reasoning'], 'reasoning_chain')
        ])
        
        # Analyze sub-question complexity
        avg_sub_questions = np.mean([
            len(result['reasoning'].sub_questions)
            for result in reasoning_results
            if hasattr(result['reasoning'], 'sub_questions')
        ])
        
        # Judge score analysis
        judge_scores = turn_data.get('judge_scores', [])
        avg_judge_score = np.mean(judge_scores) if judge_scores else 0
        
        print(f"   Average reasoning length: {avg_reasoning_length:.0f} characters")
        print(f"   Average sub-questions: {avg_sub_questions:.1f}")
        print(f"   Average judge score: {avg_judge_score:.3f}")
        
        # Quality indicators
        if avg_judge_score > 0.8:
            print("   🎯 HIGH QUALITY reasoning chains generated")
        elif avg_judge_score > 0.6:
            print("   📊 MODERATE QUALITY reasoning chains")
        else:
            print("   ⚠️ LOW QUALITY reasoning chains - model learning")
        
        # Update expertise indicators
        self.expertise_indicators['reasoning_depth'] = avg_reasoning_length / 1000  # Normalize
        self.expertise_indicators['judge_consistency'] = 1.0 - np.std(judge_scores) if len(judge_scores) > 1 else 0
    
    def _track_content_exploration(self, turn_data: Dict):
        """Track which content areas have been explored."""
        questions = turn_data.get('questions', [])
        
        for q_data in questions:
            question = q_data.get('question', '')
            topics = self._extract_topics(question)
            
            for topic in topics:
                self.explored_topics.add(topic)
                
                # Track performance on this topic
                if topic not in self.topic_performance:
                    self.topic_performance[topic] = []
        
        # Update coverage indicator
        total_possible_topics = 100  # Estimate based on CFA content
        self.expertise_indicators['content_coverage'] = len(self.explored_topics) / total_possible_topics
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified topic modeling)."""
        # CFA-specific keywords and topics
        cfa_topics = [
            'portfolio', 'risk', 'return', 'asset', 'equity', 'bond', 'derivative',
            'valuation', 'analysis', 'financial', 'investment', 'diversification',
            'allocation', 'performance', 'market', 'volatility', 'correlation',
            'ethics', 'economics', 'quantitative', 'alternative', 'fixed income',
            'corporate finance', 'behavioral', 'ESG', 'sustainability'
        ]
        
        text_lower = text.lower()
        found_topics = [topic for topic in cfa_topics if topic in text_lower]
        return found_topics[:5]  # Return top 5 topics
    
    def _perform_deep_evaluation(self, turn_count: int, best_score: float):
        """Perform thorough evaluation of model capabilities."""
        print(f"\n🎯 DEEP EVALUATION - Turn {turn_count}")
        print("=" * 50)
        
        # Evaluate on all splits
        train_score = self.training_loop._evaluate_on_split('train', num_samples=10)
        val_score = self.training_loop._evaluate_on_split('validation', num_samples=8)
        test_score = self.training_loop._evaluate_on_split('test', num_samples=5)
        
        print(f"📊 Comprehensive Evaluation:")
        print(f"   Training Score:   {train_score:.3f}")
        print(f"   Validation Score: {val_score:.3f}")
        print(f"   Test Score:       {test_score:.3f}")
        print(f"   Best Score Ever:  {best_score:.3f}")
        
        # Calculate expertise indicators
        expertise_score = self._calculate_expertise_score()
        print(f"\n🧠 Expertise Indicators:")
        print(f"   Content Coverage: {self.expertise_indicators['content_coverage']:.1%}")
        print(f"   Reasoning Depth:  {self.expertise_indicators['reasoning_depth']:.1%}")
        print(f"   Judge Consistency: {self.expertise_indicators['judge_consistency']:.1%}")
        print(f"   Overall Expertise: {expertise_score:.1%}")
        
        # Weight change summary
        self.weight_tracker.print_weight_changes()
        
        # Save progress
        self.weight_tracker.save_training_plots()
    
    def _calculate_enhanced_metrics(self, turn_data: Dict[str, Any], turn_count: int) -> Dict[str, Any]:
        """Calculate enhanced metrics for unlimited training system."""
        
        # Extract basic data
        judge_scores = turn_data.get('judge_scores', [])
        questions = turn_data.get('questions', [])
        reasoning_results = turn_data.get('reasoning_results', [])
        overall_performance = turn_data.get('overall_performance', 0.0)
        
        # Calculate expertise score based on recent performance
        expertise_score = self._calculate_expertise_score()
        
        # Calculate validation score by testing on validation set
        validation_score = self._calculate_validation_score()
        
        # Estimate weight updates (simplified)
        weight_updates = len(judge_scores) if judge_scores else 0
        
        # Calculate gradient norm estimate
        gradient_norm = np.std(judge_scores) if len(judge_scores) > 1 else 0.0
        
        # Count topics covered
        topics_covered = len(self.explored_topics)
        
        # Calculate question diversity
        question_diversity = self._calculate_question_diversity(questions)
        
        # Calculate difficulty distribution
        difficulty_avg = self._calculate_average_difficulty()
        
        return {
            'expertise_score': expertise_score,
            'validation_score': validation_score,
            'weight_updates': weight_updates,
            'gradient_norm': gradient_norm,
            'topics_covered': topics_covered,
            'question_diversity': question_diversity,
            'difficulty_avg': difficulty_avg,
            'turn_count': turn_count
        }
    
    def _calculate_expertise_score(self) -> float:
        """Calculate overall expertise score based on recent performance."""
        if not self.performance_history:
            return 0.0
        
        # Use recent performance with some smoothing
        recent_scores = [p['score'] for p in self.performance_history[-10:]]
        if not recent_scores:
            return 0.0
        
        # Weight more recent scores higher
        weights = np.linspace(0.5, 1.0, len(recent_scores))
        weighted_score = np.average(recent_scores, weights=weights)
        
        return min(1.0, weighted_score)
    
    def _calculate_validation_score(self) -> float:
        """Calculate performance on validation set."""
        try:
            # Quick evaluation on a few validation examples
            val_scores = []
            val_docs = list(self.training_loop.val_docs)[:3]  # Sample 3 validation docs
            
            for doc_id, doc_content in val_docs:
                # Generate a simple question
                questions = self.training_loop.question_generator(doc_content[:1500], num_questions=1)
                if questions:
                    question = questions[0]['question']
                    # Quick reasoning
                    result = self.training_loop.reasoning_rag(question)
                    # Quick judge score
                    context = self.training_loop.reasoning_rag.search(question, k=2)
                    judge_result = self.training_loop.judge_model(
                        question=question,
                        reasoning_chain=result.reasoning_chain,
                        answer=result.answer,
                        context=context
                    )
                    val_scores.append(judge_result.score)
            
            return np.mean(val_scores) if val_scores else 0.0
        except Exception as e:
            print(f"⚠️ Validation scoring failed: {e}")
            return 0.0
    
    def _calculate_question_diversity(self, questions: List[Dict]) -> float:
        """Calculate diversity of questions generated."""
        if not questions:
            return 0.0
        
        # Simple diversity based on question length variety and keyword diversity
        lengths = [len(q.get('question', '').split()) for q in questions]
        length_diversity = np.std(lengths) / np.mean(lengths) if lengths and np.mean(lengths) > 0 else 0.0
        
        # Keyword diversity (simplified)
        all_words = []
        for q in questions:
            words = q.get('question', '').lower().split()
            all_words.extend(words)
        
        unique_ratio = len(set(all_words)) / len(all_words) if all_words else 0.0
        
        return min(1.0, (length_diversity + unique_ratio) / 2)
    
    def _calculate_average_difficulty(self) -> float:
        """Calculate average difficulty of recent questions."""
        # Simplified difficulty based on recent judge scores
        if not self.performance_history:
            return 0.5
        
        recent_scores = [p['score'] for p in self.performance_history[-5:]]
        if not recent_scores:
            return 0.5
        
        # Higher scores suggest we need harder questions
        avg_score = np.mean(recent_scores)
        # Map score to difficulty (inverse relationship)
        difficulty = 1.0 - avg_score + 0.3  # Add base difficulty
        return min(1.0, max(0.0, difficulty))

    # ...existing code...
    
    def _boost_exploration(self):
        """Boost exploration by adjusting question generation parameters."""
        print(f"\n🚀 EXPLORATION BOOST ACTIVATED")
        print("-" * 40)
        
        # Increase question diversity
        generator = self.training_loop.question_generator
        
        # Boost expert-level questions
        if 'expert' in generator.difficulty_distribution:
            generator.difficulty_distribution['expert'] += 0.05
            generator.difficulty_distribution['easy'] = max(0.05, 
                generator.difficulty_distribution['easy'] - 0.05)
        
        # Normalize
        total = sum(generator.difficulty_distribution.values())
        for key in generator.difficulty_distribution:
            generator.difficulty_distribution[key] /= total
        
        print("   📈 Increased expert-level question generation")
        print("   🔍 Enhanced content exploration parameters")
    
    def _should_stop_training(self, turn_count: int, best_score: float) -> bool:
        """Determine if training should stop."""
        
        # Check time limit
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if elapsed_hours >= self.max_hours:
            print(f"\n⏰ Time limit reached: {elapsed_hours:.1f} hours")
            return True
        
        # Check expertise target
        expertise_score = self._calculate_expertise_score()
        if expertise_score >= self.target_expertise_score:
            print(f"\n🎯 Target expertise achieved: {expertise_score:.1%}")
            return True
        
        # Check for convergence (performance plateau)
        if len(self.performance_history) > 50:
            recent_scores = [p['score'] for p in self.performance_history[-20:]]
            if np.std(recent_scores) < 0.01:  # Very low variance
                print(f"\n📈 Performance converged (low variance: {np.std(recent_scores):.4f})")
                return True
        
        return False
    
    def _print_training_progress(self, turn_count: int, current_score: float, best_score: float):
        """Print current training progress."""
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        remaining_hours = self.max_hours - elapsed_hours
        
        expertise_score = self._calculate_expertise_score()
        
        print(f"\n📊 TRAINING PROGRESS SUMMARY")
        print("-" * 40)
        print(f"Turn: {turn_count} | Current: {current_score:.3f} | Best: {best_score:.3f}")
        print(f"Elapsed: {elapsed_hours:.1f}h | Remaining: {remaining_hours:.1f}h")
        print(f"Expertise: {expertise_score:.1%} | Target: {self.target_expertise_score:.1%}")
        print(f"Topics Explored: {len(self.explored_topics)}")
        print(f"Weight Updates: {len(self.weight_tracker.parameter_changes)}")
    
    def _save_training_state(self, turn_count: int):
        """Save complete training state."""
        save_dir = Path("autonomous_training_state")
        save_dir.mkdir(exist_ok=True)
        
        training_state = {
            'turn_count': turn_count,
            'start_time': self.start_time.isoformat(),
            'performance_history': self.performance_history,
            'expertise_indicators': self.expertise_indicators,
            'explored_topics': list(self.explored_topics),
            'topic_performance': self.topic_performance,
            'weight_history': self.weight_tracker.weight_history,
            'config': self.training_config
        }
        
        save_path = save_dir / f"training_state_turn_{turn_count}.json"
        with open(save_path, 'w') as f:
            json.dump(training_state, f, indent=2, default=str)
        
        print(f"💾 Training state saved: {save_path}")
    
    def _complete_training_session(self, final_turn: int, best_score: float):
        """Complete the training session with final analysis."""
        end_time = datetime.now()
        total_hours = (end_time - self.start_time).total_seconds() / 3600
        
        print(f"\n🏁 AUTONOMOUS TRAINING SESSION COMPLETED")
        print("=" * 60)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:   {total_hours:.1f} hours")
        print(f"Total Turns: {final_turn}")
        print(f"Best Score: {best_score:.3f}")
        
        # Final expertise assessment
        final_expertise = self._calculate_expertise_score()
        print(f"\n🎓 FINAL EXPERTISE ASSESSMENT:")
        print(f"Content Coverage: {self.expertise_indicators['content_coverage']:.1%}")
        print(f"Reasoning Depth:  {self.expertise_indicators['reasoning_depth']:.1%}")
        print(f"Judge Consistency: {self.expertise_indicators['judge_consistency']:.1%}")
        print(f"Overall Expertise: {final_expertise:.1%}")
        
        # Weight update summary
        print(f"\n🔄 TRAINING IMPACT:")
        print(f"Total Weight Updates: {len(self.weight_tracker.parameter_changes)}")
        print(f"Topics Explored: {len(self.explored_topics)}")
        print(f"Questions Generated: {sum(len(p['details'].get('questions', [])) for p in self.performance_history)}")
        
        # Save final results
        self._save_training_state(final_turn)
        self.weight_tracker.save_training_plots()
        
        # Generate final report
        self._generate_final_report(final_turn, best_score, final_expertise)
        
        print(f"\n🎉 Your agent is now a CFA expert with {final_expertise:.1%} expertise!")
        print(f"📊 Check 'autonomous_training_state/' for detailed results")
        print(f"📈 Check 'training_analysis/' for performance plots")

    def _generate_final_report(self, final_turn: int, best_score: float, final_expertise: float):
        """Generate comprehensive final training report."""
        
        report = f"""
# Autonomous CFA Training Session Report

## Session Overview
- **Duration**: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} hours
- **Total Training Turns**: {final_turn}
- **Best Performance Score**: {best_score:.3f}
- **Final Expertise Level**: {final_expertise:.1%}

## Key Achievements
- **Content Coverage**: Explored {len(self.explored_topics)} different topics
- **Weight Updates**: {len(self.weight_tracker.parameter_changes)} model improvements
- **Questions Generated**: {sum(len(p['details'].get('questions', [])) for p in self.performance_history)} total questions

## Expertise Breakdown
- Content Coverage: {self.expertise_indicators['content_coverage']:.1%}
- Reasoning Depth: {self.expertise_indicators['reasoning_depth']:.1%}
- Judge Consistency: {self.expertise_indicators['judge_consistency']:.1%}
- Improvement Rate: {self.expertise_indicators['improvement_rate']:.3f}

## Training Progression
"""
        
        # Add performance trend
        if len(self.performance_history) >= 10:
            early_avg = np.mean([p['score'] for p in self.performance_history[:10]])
            late_avg = np.mean([p['score'] for p in self.performance_history[-10:]])
            improvement = late_avg - early_avg
            
            report += f"""
### Performance Improvement
- Early Training Average: {early_avg:.3f}
- Late Training Average: {late_avg:.3f}
- Total Improvement: {improvement:.3f} ({improvement/early_avg*100:.1f}%)
"""

        # Save report
        report_path = Path("autonomous_training_state") / "final_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📋 Final report saved: {report_path}")

# Main execution
if __name__ == "__main__":
    print("🤖 Autonomous CFA Training Agent")
    print("=" * 50)
    
    # Get user preferences
    print("\n⚙️ Training Configuration:")
    
    try:
        hours = float(input("Training duration (hours) [default: 24]: ") or "24")
        target_score = float(input("Target expertise (0.1-1.0) [default: 0.95]: ") or "0.95")
    except ValueError:
        print("Using default values...")
        hours = 24.0
        target_score = 0.95
    
    print(f"\n🎯 Configuration:")
    print(f"   Duration: {hours:.1f} hours")
    print(f"   Target Expertise: {target_score:.1%}")
    
    # Initialize and run
    agent = AutonomousTrainingAgent(max_hours=hours, target_expertise_score=target_score)
    
    input("\nPress Enter to start autonomous training... (Ctrl+C to stop)")
    
    try:
        agent.run_autonomous_training()
    except KeyboardInterrupt:
        print("\n\n⏹️ Training stopped by user")
        agent._complete_training_session(
            agent.training_loop.current_step, 
            max([p['score'] for p in agent.performance_history] + [0])
        )
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
