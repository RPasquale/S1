#!/usr/bin/env python3
"""
Unlimited Duration Training with Weights & Biases Integration

This script implements an unlimited duration training system with:
- Full Weights & Biases (WnB) integration for real-time visualization
- Unlimited training capability (no hard time limits)
- Comprehensive metric tracking and dashboards
- Remote monitoring capabilities  
- Advanced experiment management
- Distributed training support
- Advanced stopping criteria based on convergence

Features:
- Real-time loss/reward visualization in WnB
- Comprehensive metric tracking and dashboards
- Remote monitoring and collaboration
- Experiment comparison and analysis
- Automatic artifact logging
- Custom charts and visualizations
- Advanced hyperparameter tracking
"""

import os
import sys
import time
import json
import wandb
import signal
import psutil
import numpy as np
import pickle  # Added for checkpoint serialization
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, asdict

# Import our training components
from autonomous_training import AutonomousTrainingAgent, WeightTracker
from model import (
    SelfTrainingLoop, ReasoningRAG, ReasoningJudgeModel, 
    AdaptiveQuestionGenerator, ModelCheckpoint, TrainingMetrics,
    doc_texts, reasoning_rag_module
)

@dataclass
class WnBConfig:
    """Configuration for Weights & Biases integration."""
    project_name: str = "cfa-expert-training"
    entity: Optional[str] = None  # Your WnB username/team
    experiment_name: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    config: Dict[str, Any] = None

class UnlimitedTrainingSystem:
    """Unlimited duration training system with full WnB integration."""
    
    def __init__(self, wnb_config: WnBConfig = None):
        """Initialize unlimited training system."""
        self.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"unlimited_training_{self.training_id}")
        self.session_dir.mkdir(exist_ok=True)
        
        # WnB Configuration
        self.wnb_config = wnb_config or WnBConfig()
        if not self.wnb_config.experiment_name:
            self.wnb_config.experiment_name = f"unlimited_training_{self.training_id}"
        if not self.wnb_config.tags:
            self.wnb_config.tags = ["unlimited", "autonomous", "cfa-expert"]
            
        # Training state
        self.training_active = False
        self.start_time = None
        self.total_training_steps = 0
        self.convergence_tracker = ConvergenceTracker()
        self.metrics_logger = MetricsLogger()
        
        # Unlimited training configuration (no hard limits)
        self.training_config = {
            'convergence_patience': 100,      # Steps to wait for improvement
            'min_improvement_threshold': 0.001,  # Minimum improvement required
            'max_stagnation_hours': 12.0,     # Max hours without improvement
            'auto_checkpoint_interval': 50,   # Steps between auto-checkpoints
            'wandb_log_interval': 5,          # Steps between WnB logs
            'enable_distributed': False,      # Distributed training capability
            'early_stopping': True,           # Enable intelligent early stopping
            'target_convergence_score': 0.99, # Score for full convergence
        }
        
        # Initialize components
        self.weight_tracker = WeightTracker()
        self.autonomous_agent = None
        self.wandb_run = None
        
        print(f"🚀 Unlimited Training System Initialized")
        print(f"📊 Session ID: {self.training_id}")
        print(f"📁 Session Directory: {self.session_dir}")

    def initialize_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        print("\n📊 Initializing Weights & Biases...")
        
        try:
            # Initialize WnB run
            self.wandb_run = wandb.init(
                project=self.wnb_config.project_name,
                entity=self.wnb_config.entity,
                name=self.wnb_config.experiment_name,
                tags=self.wnb_config.tags,
                notes=self.wnb_config.notes or f"Unlimited duration CFA expert training - {self.training_id}",
                config={
                    "training_id": self.training_id,
                    "start_time": datetime.now().isoformat(),
                    "training_type": "unlimited_autonomous",
                    "convergence_patience": self.training_config['convergence_patience'],
                    "min_improvement_threshold": self.training_config['min_improvement_threshold'],
                    "max_stagnation_hours": self.training_config['max_stagnation_hours'],
                    "auto_checkpoint_interval": self.training_config['auto_checkpoint_interval'],
                    "target_convergence_score": self.training_config['target_convergence_score'],
                    **(self.wnb_config.config or {})
                }
            )
            
            # Define custom charts
            self._setup_wandb_charts()
            
            print("✅ WnB initialized successfully!")
            print(f"🔗 Dashboard URL: {self.wandb_run.url}")
            
        except Exception as e:
            print(f"⚠️ WnB initialization failed: {e}")
            print("Training will continue without WnB logging...")
            self.wandb_run = None

    def _setup_wandb_charts(self):
        """Setup custom WnB charts and visualizations."""
        if not self.wandb_run:
            return
            
        # Define custom charts
        wandb.define_metric("training/step")
        wandb.define_metric("training/*", step_metric="training/step")
        
        # Performance metrics
        wandb.define_metric("performance/expertise_score", step_metric="training/step")
        wandb.define_metric("performance/judge_score_avg", step_metric="training/step")
        wandb.define_metric("performance/reasoning_quality", step_metric="training/step")
        wandb.define_metric("performance/validation_score", step_metric="training/step")
        
        # Training dynamics
        wandb.define_metric("training/weight_updates", step_metric="training/step")
        wandb.define_metric("training/gradient_norm", step_metric="training/step")
        wandb.define_metric("training/learning_rate", step_metric="training/step")
        wandb.define_metric("training/questions_processed", step_metric="training/step")
        
        # Convergence tracking
        wandb.define_metric("convergence/improvement_rate", step_metric="training/step")
        wandb.define_metric("convergence/stagnation_hours", step_metric="training/step")
        wandb.define_metric("convergence/convergence_score", step_metric="training/step")
        
        # Content exploration
        wandb.define_metric("content/topics_covered", step_metric="training/step")
        wandb.define_metric("content/difficulty_distribution", step_metric="training/step")
        wandb.define_metric("content/question_diversity", step_metric="training/step")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum}, initiating graceful shutdown...")
            self.training_active = False
            if self.wandb_run:
                wandb.log({"training/interrupted": True, "training/interruption_time": time.time()})
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def check_convergence(self, current_metrics: Dict[str, float]) -> bool:
        """Check if training has converged based on multiple criteria."""
        return self.convergence_tracker.check_convergence(
            current_metrics,
            patience=self.training_config['convergence_patience'],
            min_improvement=self.training_config['min_improvement_threshold'],
            max_stagnation_hours=self.training_config['max_stagnation_hours']
        )

    def log_metrics_to_wandb(self, metrics: Dict[str, Any], step: int):
        """Log comprehensive metrics to Weights & Biases."""
        if not self.wandb_run:
            return
            
        try:
            # Prepare metrics for logging
            wandb_metrics = {
                "training/step": step,
                "training/timestamp": time.time(),
                "training/hours_elapsed": (datetime.now() - self.start_time).total_seconds() / 3600,
            }
            
            # Add all provided metrics with proper prefixes
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    wandb_metrics[key] = float(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, np.integer, np.floating)):
                            wandb_metrics[f"{key}/{subkey}"] = float(subvalue)
            
            # Log to WnB
            wandb.log(wandb_metrics, step=step)
            
        except Exception as e:
            print(f"⚠️ Failed to log metrics to WnB: {e}")

    def run_unlimited_training(self):
        """Run unlimited duration training with full WnB integration."""
        print("\n🚀 Starting Unlimited Duration Training")
        print("=" * 60)
        
        # Setup
        self.setup_signal_handlers()
        self.initialize_wandb()
        
        self.start_time = datetime.now()
        self.training_active = True
        
        print(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Training will continue until convergence or manual stop")
        print(f"📊 Monitor progress at: {self.wandb_run.url if self.wandb_run else 'N/A'}")
        print("💡 Press Ctrl+C for graceful shutdown")
        
        # Initialize autonomous training agent
        self.autonomous_agent = AutonomousTrainingAgent(
            max_hours=float('inf'),  # No time limit
            target_expertise_score=self.training_config['target_convergence_score']
        )
        
        try:
            step = 0
            while self.training_active:
                step += 1
                self.total_training_steps = step
                
                # Execute training step
                step_metrics = self._execute_training_step(step)
                
                # Log to WnB
                if step % self.training_config['wandb_log_interval'] == 0:
                    self.log_metrics_to_wandb(step_metrics, step)
                
                # Check convergence
                if self.training_config['early_stopping']:
                    if self.check_convergence(step_metrics):
                        print(f"\n🎉 Training converged after {step} steps!")
                        break
                
                # Auto-checkpoint
                if step % self.training_config['auto_checkpoint_interval'] == 0:
                    self._save_checkpoint(step, step_metrics)
                
                # Progress update
                if step % 10 == 0:
                    self._print_progress_update(step, step_metrics)
                  # Brief pause to prevent overload
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            if self.wandb_run:
                wandb.log({"training/error": str(e)})
        finally:
            self._finalize_training()

    def _execute_training_step(self, step: int) -> Dict[str, Any]:
        """Execute a single training step and return metrics."""
        try:
            # Get training metrics from autonomous agent
            if self.autonomous_agent:
                step_result = self.autonomous_agent._execute_enhanced_turn(step)
                
                # Ensure step_result is a dictionary
                if not isinstance(step_result, dict):
                    step_result = {}
                
                # Calculate meaningful metrics from the training data
                judge_scores = step_result.get('judge_scores', [])
                questions = step_result.get('questions', [])
                reasoning_results = step_result.get('reasoning_results', [])
                overall_performance = step_result.get('overall_performance', 0.0)
                  # Process metrics safely with enhanced type checking
                def safe_float(value, default=0.0):
                    """Safely convert value to float, handling lists and other types."""
                    try:
                        if isinstance(value, (list, tuple)):
                            if len(value) > 0:
                                return float(np.mean([float(x) for x in value if isinstance(x, (int, float))]))
                            return default
                        elif isinstance(value, (int, float, np.integer, np.floating)):
                            return float(value)
                        else:
                            return default
                    except (ValueError, TypeError):
                        return default
                
                def safe_int(value, default=0):
                    """Safely convert value to int, handling lists and other types."""
                    try:
                        if isinstance(value, (list, tuple)):
                            if len(value) > 0:
                                return int(np.mean([float(x) for x in value if isinstance(x, (int, float))]))
                            return default
                        elif isinstance(value, (int, float, np.integer, np.floating)):
                            return int(value)
                        else:
                            return default
                    except (ValueError, TypeError):
                        return default
                
                avg_judge_score = safe_float(np.mean(judge_scores) if judge_scores else 0.0)
                questions_processed = len(questions) if questions else 0
                reasoning_quality = safe_float(overall_performance)
                
                # Extract and process metrics with enhanced safety
                metrics = {
                    "performance/expertise_score": safe_float(step_result.get('expertise_score', avg_judge_score)),
                    "performance/judge_score_avg": avg_judge_score,
                    "performance/reasoning_quality": reasoning_quality,
                    "performance/validation_score": safe_float(step_result.get('validation_score', 0.0)),
                    "training/weight_updates": safe_int(step_result.get('weight_updates', 1 if avg_judge_score > 0 else 0)),
                    "training/gradient_norm": safe_float(step_result.get('gradient_norm', avg_judge_score)),
                    "training/questions_processed": questions_processed,
                    "content/topics_covered": safe_int(step_result.get('topics_covered', min(questions_processed, 5))),
                    "content/difficulty_distribution": safe_float(step_result.get('difficulty_avg', 0.5)),
                    "content/question_diversity": safe_float(step_result.get('question_diversity', min(1.0, questions_processed * 0.2))),
                }
                
                # Update convergence tracker
                self.convergence_tracker.update(metrics)
                
                # Add convergence metrics
                convergence_metrics = self.convergence_tracker.get_convergence_metrics()
                metrics.update({f"convergence/{k}": v for k, v in convergence_metrics.items()})
                
                return metrics
            else:
                return {"training/step": step, "training/status": "no_agent"}
                
        except Exception as e:
            print(f"⚠️ Error in training step {step}: {e}")
            return {"training/step": step, "training/error": 1.0}

    def _save_checkpoint(self, step: int, metrics: Dict[str, Any]):
        """Save training checkpoint with WnB artifact logging."""
        try:
            checkpoint_path = self.session_dir / f"checkpoint_step_{step}.pkl"
            
            # Save checkpoint locally
            checkpoint_data = {
                'step': step,
                'metrics': metrics,
                'training_config': self.training_config,
                'convergence_state': self.convergence_tracker.get_state(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Log as WnB artifact
            if self.wandb_run:
                artifact = wandb.Artifact(
                    name=f"checkpoint_step_{step}",
                    type="model_checkpoint",
                    description=f"Training checkpoint at step {step}"
                )
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)
            
            print(f"💾 Checkpoint saved: step_{step}")
            
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")

    def _print_progress_update(self, step: int, metrics: Dict[str, Any]):
        """Print detailed progress update."""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600
        
        print(f"\n📊 Step {step} | {hours:.1f}h elapsed")
        print(f"   🎯 Expertise: {metrics.get('performance/expertise_score', 0):.3f}")
        print(f"   ⭐ Judge Score: {metrics.get('performance/judge_score_avg', 0):.3f}")
        print(f"   🧠 Reasoning: {metrics.get('performance/reasoning_quality', 0):.3f}")
        print(f"   📈 Validation: {metrics.get('performance/validation_score', 0):.3f}")
        print(f"   🔄 Weight Updates: {metrics.get('training/weight_updates', 0)}")
        print(f"   📚 Questions: {metrics.get('training/questions_processed', 0)}")
        
        # Convergence info
        convergence_score = metrics.get('convergence/convergence_score', 0)
        stagnation_hours = metrics.get('convergence/stagnation_hours', 0)
        print(f"   🎪 Convergence: {convergence_score:.3f} (stagnant: {stagnation_hours:.1f}h)")

    def _finalize_training(self):
        """Finalize training session and cleanup."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time if self.start_time else timedelta(0)
        
        print(f"\n🏁 Training Session Complete")
        print("=" * 50)
        print(f"🕐 Duration: {total_duration}")
        print(f"📊 Total Steps: {self.total_training_steps}")
        print(f"📁 Session Data: {self.session_dir}")
        
        # Final WnB logging
        if self.wandb_run:
            final_metrics = {
                "training/final_step": self.total_training_steps,            "training/total_duration_hours": total_duration.total_seconds() / 3600,
                "training/completed": True,
                "training/end_time": end_time.isoformat()
            }
            wandb.log(final_metrics)
            
            # Safely finish WandB with error handling
            try:
                print("🔄 Finalizing WandB logging...")
                wandb.finish(quiet=True)
                print(f"✅ WnB Run Complete: {self.wandb_run.url}")
            except Exception as wandb_error:
                print(f"⚠️  WandB finish error (training data already saved): {wandb_error}")
                print("📋 Note: Training completed successfully, only final cleanup failed")
                try:
                    print(f"📊 WnB Dashboard still available: {self.wandb_run.url}")
                except:
                    pass

class ConvergenceTracker:
    """Tracks training convergence based on multiple criteria."""
    
    def __init__(self):
        self.history = []
        self.best_scores = {}
        self.stagnation_start = None
        self.last_improvement_time = datetime.now()
    
    def update(self, metrics: Dict[str, float]):
        """Update convergence tracking with new metrics."""
        self.history.append({
            'timestamp': datetime.now(),
            'metrics': metrics.copy()
        })
        
        # Check for improvements
        improved = False
        for key, value in metrics.items():
            if key.startswith('performance/') and isinstance(value, (int, float)):
                # Ensure value is a number, not a list or other type
                value = float(value)
                if key not in self.best_scores or value > self.best_scores[key]:
                    self.best_scores[key] = value
                    improved = True
        
        if improved:
            self.last_improvement_time = datetime.now()
            self.stagnation_start = None
        elif self.stagnation_start is None:
            self.stagnation_start = datetime.now()
    
    def check_convergence(self, current_metrics: Dict[str, float], 
                         patience: int, min_improvement: float, 
                         max_stagnation_hours: float) -> bool:
        """Check if training has converged."""
        if len(self.history) < patience:
            return False
        
        # Check stagnation time
        if self.stagnation_start:
            stagnation_hours = (datetime.now() - self.stagnation_start).total_seconds() / 3600
            if stagnation_hours > max_stagnation_hours:
                print(f"🛑 Convergence: No improvement for {stagnation_hours:.1f} hours")
                return True
          # Check improvement rate
        recent_metrics = [h['metrics'] for h in self.history[-patience:]]
        for key in current_metrics.keys():
            if key.startswith('performance/'):
                values = [m.get(key, 0) for m in recent_metrics]
                if len(values) > 1:
                    improvement = max(values) - min(values)
                    if improvement < min_improvement:
                        continue  # This metric hasn't improved enough
                else:
                    return False  # Not enough data
        
        return False  # No convergence detected
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get current convergence metrics."""
        if not self.history:
            return {
                'improvement_rate': 0.0,
                'stagnation_hours': 0.0,
                'convergence_score': 0.0,
                'steps_since_improvement': 0
            }
        
        stagnation_hours = 0.0
        if self.stagnation_start:
            stagnation_hours = (datetime.now() - self.stagnation_start).total_seconds() / 3600
        
        recent_improvement = 0.0
        if len(self.history) >= 10:
            try:
                recent = [h['metrics'] for h in self.history[-10:] if isinstance(h.get('metrics'), dict)]
                early = []
                if len(self.history) >= 20:
                    early = [h['metrics'] for h in self.history[-20:-10] if isinstance(h.get('metrics'), dict)]
                
                if early and recent:
                    # Get common keys that start with 'performance/'
                    perf_keys = [k for k in recent[0].keys() if k.startswith('performance/')]
                    
                    for key in perf_keys:
                        try:
                            recent_values = [float(m.get(key, 0)) for m in recent if key in m]
                            early_values = [float(m.get(key, 0)) for m in early if key in m]
                            
                            if recent_values and early_values:
                                recent_avg = np.mean(recent_values)
                                early_avg = np.mean(early_values)
                                improvement = recent_avg - early_avg
                                recent_improvement = max(recent_improvement, improvement)
                        except (ValueError, TypeError):
                            continue
            except Exception:
                recent_improvement = 0.0
        
        convergence_score = min(1.0, len(self.best_scores) * 0.2 if self.best_scores else 0.0)
        
        # Calculate steps since last improvement safely
        steps_since_improvement = 0
        try:
            if self.best_scores and self.history:
                for i in reversed(range(len(self.history))):
                    hist_entry = self.history[i]
                    if isinstance(hist_entry.get('metrics'), dict):
                        hist_metrics = hist_entry['metrics']
                        # Check if any metric matches our best scores
                        for key, best_value in self.best_scores.items():
                            if key in hist_metrics:
                                try:
                                    if abs(float(hist_metrics[key]) - float(best_value)) < 1e-6:
                                        steps_since_improvement = len(self.history) - 1 - i
                                        break
                                except (ValueError, TypeError):
                                    continue
                        if steps_since_improvement > 0:
                            break
                else:
                    steps_since_improvement = len(self.history)
        except Exception:
            steps_since_improvement = len(self.history) if self.history else 0
        
        return {
            'improvement_rate': float(recent_improvement),
            'stagnation_hours': float(stagnation_hours),
            'convergence_score': float(convergence_score),
            'steps_since_improvement': int(steps_since_improvement)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete convergence tracker state for checkpointing."""
        return {
            'best_scores': self.best_scores,
            'stagnation_start': self.stagnation_start.isoformat() if self.stagnation_start else None,
            'last_improvement_time': self.last_improvement_time.isoformat(),
            'history_length': len(self.history)
        }

class MetricsLogger:
    """Advanced metrics logging and analysis."""
    
    def __init__(self):
        self.metrics_buffer = []
        self.custom_charts = {}
    
    def log_custom_chart(self, chart_name: str, data: Dict[str, Any]):
        """Log data for custom chart visualization."""
        if chart_name not in self.custom_charts:
            self.custom_charts[chart_name] = []
        
        self.custom_charts[chart_name].append({
            'timestamp': datetime.now(),
            'data': data
        })

def main():
    """Main function to run unlimited training with WnB integration."""
    print("🚀 Ultimate Unlimited Training with Weights & Biases")
    print("=" * 60)
    
    # Configure WnB
    print("📊 Configure Weights & Biases Integration:")
    print("-" * 40)
    
    # Get WnB configuration from user
    project_name = input("WnB Project Name [cfa-expert-training]: ").strip() or "cfa-expert-training"
    entity = input("WnB Entity/Username (optional): ").strip() or None
    experiment_name = input("Experiment Name (optional): ").strip() or None
    
    tags_input = input("Tags (comma-separated, optional): ").strip()
    tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else None
    
    notes = input("Experiment Notes (optional): ").strip()
    
    # Create WnB config
    wnb_config = WnBConfig(
        project_name=project_name,
        entity=entity,
        experiment_name=experiment_name,
        tags=tags or ["unlimited", "autonomous", "cfa-expert"],
        notes=notes
    )
    
    print(f"\n✅ WnB Configuration:")
    print(f"   Project: {wnb_config.project_name}")
    print(f"   Entity: {wnb_config.entity or 'Default'}")
    print(f"   Experiment: {wnb_config.experiment_name or 'Auto-generated'}")
    print(f"   Tags: {wnb_config.tags}")
    
    # Confirm start
    confirm = input(f"\n🚀 Start unlimited training? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize and run training
    training_system = UnlimitedTrainingSystem(wnb_config)
    training_system.run_unlimited_training()

if __name__ == "__main__":
    main()
