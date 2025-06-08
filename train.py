#!/usr/bin/env python3
"""
Unified Document Training System
Consolidates all training functionality into one comprehensive script with
automatic device detection, argument parsing, and WandB integration.

This script replaces all individual training scripts with a unified interface
supporting all training modes: Maximum GPU, Intensive, Continuous, Quick Test,
Custom, Unlimited, and Autonomous training.
"""

import argparse
import os
import sys
import time
import datetime
import json
import threading
import signal
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training components
from model import self_training_system, OptimizedDualTrainer
from config import DOCUMENTS_FOLDER, INDEX_FOLDER

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not available. Install with: pip install wandb")

# GPU monitoring (optional)
try:
    import pynvml
    import torch
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

class UnifiedTrainer:
    """Unified training system consolidating all training modes."""
    
    def __init__(self, args):
        self.args = args
        self.device = self._detect_device()
        self.start_time = None
        self.session_dir = None
        self.wandb_run = None
        self.interrupted = False
        
        # Performance tracking
        self.performance_metrics = {
            'turns_completed': 0,
            'questions_processed': 0,
            'best_ntp_loss': float('inf'),
            'best_rl_score': 0.0,
            'gpu_metrics': [],
            'training_history': []
        }
        
        # Initialize trainer
        self.trainer = OptimizedDualTrainer(
            ntp_lr=args.ntp_lr,
            rl_lr=args.rl_lr,
            device=self.device
        )
        
        self._setup_session()
        self._setup_logging()
        self._setup_signal_handlers()
        
        if args.wandb and WANDB_AVAILABLE:
            self._setup_wandb()

    def _detect_device(self):
        """Automatically detect the best available device."""
        try:
            import torch
            
            # Check for CUDA
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🚀 CUDA Device Detected: {device_name} ({memory_gb:.1f} GB)")
                return 'cuda'
            
            # Check for MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🍎 MPS Device Detected (Apple Silicon)")
                return 'mps'
            
            else:
                print("💻 Using CPU (No GPU acceleration available)")
                return 'cpu'
                
        except ImportError:
            print("⚠️ PyTorch not available, using CPU")
            return 'cpu'

    def _setup_session(self):
        """Setup training session directory and files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"training_session_{self.args.mode}_{timestamp}")
        self.session_dir.mkdir(exist_ok=True)
        
        print(f"📁 Training Session: {self.session_dir}")

    def _setup_logging(self):
        """Setup logging for the training session."""
        log_file = self.session_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⏸️ Training interrupted (Signal {signum})")
            self.interrupted = True
            self._finalize_training()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_wandb(self):
        """Setup Weights & Biases integration."""
        if not WANDB_AVAILABLE:
            return
        
        project_name = f"document-{self.args.mode}-training"
        run_name = f"{self.args.mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        
        config = {
            "mode": self.args.mode,
            "device": self.device,
            "ntp_lr": self.args.ntp_lr,
            "rl_lr": self.args.rl_lr,
            "hours": getattr(self.args, 'hours', None),
            "questions_per_turn": self.args.questions_per_turn,
            "batch_size": self.args.batch_size,
            "memory_threshold": self.args.memory_threshold,
            "checkpoint_frequency": self.args.checkpoint_freq,
        }
        
        self.wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=[self.args.mode, "document-expert", self.device]
        )
        
        # Setup custom metrics
        wandb.define_metric("training/step")
        wandb.define_metric("training/*", step_metric="training/step")
        wandb.define_metric("ntp/*", step_metric="training/step")
        wandb.define_metric("rl/*", step_metric="training/step")
        wandb.define_metric("gpu/*", step_metric="training/step")
        wandb.define_metric("system/*", step_metric="training/step")
        
        print(f"🌐 WandB Dashboard: {wandb.run.url}")

    def _get_gpu_metrics(self):
        """Get comprehensive GPU metrics."""
        if not GPU_MONITORING_AVAILABLE or self.device != 'cuda':
            return {}
        
        try:
            # PyTorch metrics
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_percent = (memory_allocated / memory_total) * 100
            
            metrics = {
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved,
                "gpu_memory_total_gb": memory_total,
                "gpu_memory_percent": memory_percent,
            }
            
            # NVIDIA-ML metrics
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_utilization"] = util.gpu
                metrics["gpu_memory_utilization"] = util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature"] = temp
                
                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                metrics["gpu_power_watts"] = power
                
            except Exception:
                pass  # NVIDIA-ML metrics optional
                
            return metrics
            
        except Exception as e:
            self.logger.warning(f"GPU metrics error: {e}")
            return {}

    def _log_training_step(self, step_data):
        """Log training step to WandB and files."""
        step = self.performance_metrics['turns_completed']
        
        # Extract metrics
        ntp_loss = step_data.get('avg_ntp_loss', 0.0)
        rl_score = step_data.get('avg_rl_score', 0.0)
        questions = step_data.get('questions_generated', 0)
        
        # Update performance tracking
        self.performance_metrics['questions_processed'] += questions
        if ntp_loss < self.performance_metrics['best_ntp_loss']:
            self.performance_metrics['best_ntp_loss'] = ntp_loss
        if rl_score > self.performance_metrics['best_rl_score']:
            self.performance_metrics['best_rl_score'] = rl_score
        
        # Get system metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Log to WandB
        if self.wandb_run:
            log_data = {
                "training/step": step,
                "training/questions_total": self.performance_metrics['questions_processed'],
                "ntp/loss": ntp_loss,
                "rl/score": rl_score,
                "training/elapsed_hours": (time.time() - self.start_time) / 3600,
                **gpu_metrics
            }
            wandb.log(log_data)
        
        # Add to history
        self.performance_metrics['training_history'].append({
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "ntp_loss": ntp_loss,
            "rl_score": rl_score,
            "questions": questions,
            **gpu_metrics
        })
        
        # Log to console
        if step % 10 == 0:
            elapsed = (time.time() - self.start_time) / 3600
            print(f"📊 Step {step} | {elapsed:.1f}h | NTP: {ntp_loss:.4f} | RL: {rl_score:.3f} | Q: {questions}")

    def _save_checkpoint(self, step, additional_data=None):
        """Save training checkpoint."""
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "performance_metrics": self.performance_metrics,
            "args": vars(self.args),
            "trainer_state": "optimized_trainer_state",  # Would save actual state
            "additional_data": additional_data or {}
        }
        
        checkpoint_file = self.session_dir / f"checkpoint_step_{step}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Save trainer state
        try:
            trainer_file = self.session_dir / f"trainer_state_step_{step}.pkl"
            self.trainer.save_optimized_state(str(trainer_file))
            print(f"💾 Checkpoint saved: Step {step}")
        except Exception as e:
            self.logger.warning(f"Trainer state save failed: {e}")

    def _manage_gpu_memory(self):
        """Manage GPU memory to prevent OOM errors."""
        if self.device != 'cuda' or not GPU_MONITORING_AVAILABLE:
            return
        
        try:
            memory_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
            
            if memory_used > self.args.memory_threshold:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"🧹 GPU memory cleared: {memory_used*100:.1f}% usage")
                
        except Exception as e:
            self.logger.warning(f"GPU memory management error: {e}")

    def _train_step(self, doc_sample, step_num):
        """Execute one training step."""
        try:
            # Execute training step
            step_data = self.trainer.train_step(
                doc_sample,
                num_questions=self.args.questions_per_turn
            )
            
            # Calculate metrics
            ntp_losses = step_data.get('ntp_results', {}).get('losses', [])
            judge_scores = step_data.get('judge_scores', [])
            
            avg_ntp_loss = sum(ntp_losses) / len(ntp_losses) if ntp_losses else 1.0
            avg_rl_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
            
            processed_data = {
                'avg_ntp_loss': avg_ntp_loss,
                'avg_rl_score': avg_rl_score,
                'questions_generated': len(judge_scores),
                'step_data': step_data
            }
            
            self.performance_metrics['turns_completed'] += 1
            self._log_training_step(processed_data)
            
            # Save checkpoint
            if self.performance_metrics['turns_completed'] % self.args.checkpoint_freq == 0:
                self._save_checkpoint(self.performance_metrics['turns_completed'])
            
            # GPU memory management
            if step_num % 10 == 0:
                self._manage_gpu_memory()
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Training step {step_num} failed: {e}")
            return None

    def run_maximum_training(self):
        """Run maximum GPU utilization training."""
        print(f"🔥 Starting Maximum GPU Training")
        print(f"   ⏱️ Duration: {self.args.hours} hours")
        print(f"   🎯 GPU Target: {self.args.memory_threshold*100}% memory")
        print(f"   ❓ Questions per turn: {self.args.questions_per_turn}")
        
        end_time = time.time() + (self.args.hours * 3600)
        
        # Set aggressive GPU settings
        if self.device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(self.args.memory_threshold)
        
        step = 0
        while time.time() < end_time and not self.interrupted:
            # Select random document
            if not self_training_system.train_docs:
                print("⚠️ No training documents available")
                break
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:3000]  # Large samples for maximum training
            
            result = self._train_step(doc_sample, step)
            if result is None:
                continue
            
            step += 1
            
            # Brief pause to prevent overheating
            if step % 100 == 0:
                time.sleep(2)

    def run_intensive_training(self):
        """Run intensive training with balanced performance."""
        print(f"🚀 Starting Intensive Training")
        print(f"   ⏱️ Duration: {self.args.hours} hours")
        print(f"   📊 Questions per turn: {self.args.questions_per_turn}")
        
        end_time = time.time() + (self.args.hours * 3600)
        step = 0
        
        while time.time() < end_time and not self.interrupted:
            if not self_training_system.train_docs:
                break
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:2500]
            
            self._train_step(doc_sample, step)
            step += 1

    def run_continuous_training(self):
        """Run continuous training with regular checkpoints."""
        print(f"🔄 Starting Continuous Training")
        print(f"   🔄 Turns: {self.args.turns}")
        print(f"   📦 Batch size: {self.args.batch_size}")
        
        for turn in range(self.args.turns):
            if self.interrupted:
                break
            
            # Process batch of documents
            batch_docs = []
            for _ in range(self.args.batch_size):
                if self_training_system.train_docs:
                    import random
                    doc_id, doc_content = random.choice(self_training_system.train_docs)
                    batch_docs.append((doc_id, doc_content[:2000]))
            
            for i, (doc_id, doc_sample) in enumerate(batch_docs):
                step = turn * self.args.batch_size + i
                self._train_step(doc_sample, step)

    def run_quick_test(self):
        """Run quick test training."""
        print(f"⚡ Starting Quick Test Training")
        print(f"   🔄 Turns: {self.args.turns}")
        
        for turn in range(self.args.turns):
            if self.interrupted:
                break
            
            if not self_training_system.train_docs:
                break
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:1000]  # Smaller samples for quick test
            
            self._train_step(doc_sample, turn)

    def run_unlimited_training(self):
        """Run unlimited training until manually stopped."""
        print(f"∞ Starting Unlimited Training")
        print(f"   📊 Questions per turn: {self.args.questions_per_turn}")
        print(f"   💾 Checkpoint frequency: {self.args.checkpoint_freq}")
        print(f"   ⏹️ Press Ctrl+C to stop")
        
        step = 0
        while not self.interrupted:
            if not self_training_system.train_docs:
                print("⚠️ No training documents available")
                break
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:2000]
            
            self._train_step(doc_sample, step)
            step += 1
            
            # Brief pause every 50 steps
            if step % 50 == 0:
                time.sleep(1)

    def run_autonomous_training(self):
        """Run autonomous training with adaptive parameters."""
        print(f"🤖 Starting Autonomous Training")
        print(f"   ⏱️ Duration: {self.args.hours} hours")
        print(f"   🧠 Adaptive parameters enabled")
        
        end_time = time.time() + (self.args.hours * 3600)
        step = 0
        
        while time.time() < end_time and not self.interrupted:
            if not self_training_system.train_docs:
                break
            
            # Adaptive question count based on performance
            adaptive_questions = self.args.questions_per_turn
            if self.performance_metrics['best_rl_score'] > 0.8:
                adaptive_questions = min(8, self.args.questions_per_turn + 2)
            elif self.performance_metrics['best_rl_score'] < 0.3:
                adaptive_questions = max(2, self.args.questions_per_turn - 1)
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:2000]
            
            # Temporarily adjust questions per turn
            original_questions = self.args.questions_per_turn
            self.args.questions_per_turn = adaptive_questions
            
            self._train_step(doc_sample, step)
            
            # Restore original setting
            self.args.questions_per_turn = original_questions
            step += 1

    def run_custom_training(self):
        """Run custom training with user-defined parameters."""
        print(f"🛠️ Starting Custom Training")
        print(f"   ⏱️ Duration: {getattr(self.args, 'hours', 'N/A')} hours")
        print(f"   🔄 Turns: {getattr(self.args, 'turns', 'N/A')}")
        print(f"   📊 Questions per turn: {self.args.questions_per_turn}")
        
        if hasattr(self.args, 'hours'):
            end_time = time.time() + (self.args.hours * 3600)
            step = 0
            
            while time.time() < end_time and not self.interrupted:
                if not self_training_system.train_docs:
                    break
                
                import random
                doc_id, doc_content = random.choice(self_training_system.train_docs)
                doc_sample = doc_content[:2000]
                
                self._train_step(doc_sample, step)
                step += 1
                
        elif hasattr(self.args, 'turns'):
            for turn in range(self.args.turns):
                if self.interrupted:
                    break
                
                if not self_training_system.train_docs:
                    break
                
                import random
                doc_id, doc_content = random.choice(self_training_system.train_docs)
                doc_sample = doc_content[:2000]
                
                self._train_step(doc_sample, turn)

    def run_training(self):
        """Main training execution method."""
        self.start_time = time.time()
        
        print(f"\n🎯 Unified Document Training System")
        print(f"=" * 60)
        print(f"📊 Mode: {self.args.mode}")
        print(f"🔧 Device: {self.device}")
        print(f"📁 Session: {self.session_dir}")
        print(f"📄 Documents: {len(self_training_system.train_docs)} available")
        
        if self.wandb_run:
            print(f"🌐 WandB: {self.wandb_run.url}")
        
        print(f"=" * 60)
        
        # Route to appropriate training method
        training_methods = {
            'maximum': self.run_maximum_training,
            'intensive': self.run_intensive_training,
            'continuous': self.run_continuous_training,
            'quick': self.run_quick_test,
            'unlimited': self.run_unlimited_training,
            'autonomous': self.run_autonomous_training,
            'custom': self.run_custom_training
        }
        
        if self.args.mode in training_methods:
            try:
                training_methods[self.args.mode]()
            except KeyboardInterrupt:
                print("\n⏸️ Training interrupted by user")
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                raise
        else:
            raise ValueError(f"Unknown training mode: {self.args.mode}")
        
        self._finalize_training()

    def _finalize_training(self):
        """Finalize training session."""
        if self.start_time is None:
            return
        
        total_duration = time.time() - self.start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        
        print(f"\n🏁 Training Session Complete!")
        print(f"=" * 60)
        print(f"   ⏱️ Duration: {hours}h {minutes}m")
        print(f"   🔄 Turns: {self.performance_metrics['turns_completed']}")
        print(f"   ❓ Questions: {self.performance_metrics['questions_processed']}")
        print(f"   🏆 Best NTP Loss: {self.performance_metrics['best_ntp_loss']:.4f}")
        print(f"   🎯 Best RL Score: {self.performance_metrics['best_rl_score']:.3f}")
        print(f"   📁 Results: {self.session_dir}")
        
        # Save final checkpoint
        self._save_checkpoint(self.performance_metrics['turns_completed'], {
            "training_completed": True,
            "total_duration_seconds": total_duration,
            "final_performance": {
                "ntp_loss": self.performance_metrics['best_ntp_loss'],
                "rl_score": self.performance_metrics['best_rl_score'],
                "turns": self.performance_metrics['turns_completed'],
                "questions": self.performance_metrics['questions_processed']
            }
        })
        
        # Save training history
        history_file = self.session_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.performance_metrics['training_history'], f, indent=2, default=str)
        
        # Finish WandB
        if self.wandb_run:
            try:
                wandb.log({
                    "final/duration_hours": total_duration / 3600,
                    "final/total_turns": self.performance_metrics['turns_completed'],
                    "final/total_questions": self.performance_metrics['questions_processed'],
                    "final/best_ntp_loss": self.performance_metrics['best_ntp_loss'],
                    "final/best_rl_score": self.performance_metrics['best_rl_score']
                })
                wandb.finish(quiet=True)
                print(f"📊 WandB session completed")
            except Exception as e:
                self.logger.warning(f"WandB finish error: {e}")


def create_parser():
    """Create argument parser for all training modes."""
    parser = argparse.ArgumentParser(
        description="Unified Document Training System - All training modes in one script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  maximum     - Maximum GPU utilization (pushes hardware to limits)
  intensive   - High-performance balanced training
  continuous  - Batch-based training with regular checkpoints
  quick       - Quick test training for validation
  unlimited   - Unlimited training until manually stopped
  autonomous  - Adaptive training with intelligent parameter adjustment
  custom      - Custom training with user-defined parameters

Examples:
  # Maximum GPU training for 8 hours
  python train.py maximum --hours 8 --questions-per-turn 8 --memory-threshold 0.95

  # Intensive training with WandB
  python train.py intensive --hours 12 --wandb --questions-per-turn 6

  # Quick test run
  python train.py quick --turns 20 --questions-per-turn 3

  # Unlimited training
  python train.py unlimited --questions-per-turn 5 --checkpoint-freq 50

  # Custom training
  python train.py custom --hours 4 --questions-per-turn 4 --memory-threshold 0.8
        """
    )
    
    # Required argument
    parser.add_argument(
        'mode',
        choices=['maximum', 'intensive', 'continuous', 'quick', 'unlimited', 'autonomous', 'custom'],
        help='Training mode to run'
    )
    
    # Duration arguments (mutually exclusive)
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        '--hours',
        type=float,
        default=8.0,
        help='Training duration in hours (default: 8.0)'
    )
    duration_group.add_argument(
        '--turns',
        type=int,
        help='Number of training turns (alternative to hours)'
    )
    
    # Training parameters
    parser.add_argument(
        '--questions-per-turn',
        type=int,
        default=5,
        help='Number of questions per training turn (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for continuous training (default: 2)'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=25,
        help='Checkpoint frequency in turns (default: 25)'
    )
    
    parser.add_argument(
        '--memory-threshold',
        type=float,
        default=0.85,
        help='GPU memory threshold (0.0-1.0, default: 0.85)'
    )
    
    # Learning rates
    parser.add_argument(
        '--ntp-lr',
        type=float,
        default=1e-4,
        help='Next Token Prediction learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--rl-lr',
        type=float,
        default=5e-5,
        help='Reinforcement Learning learning rate (default: 5e-5)'
    )
    
    # Integration options
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpoint saving'
    )
    
    # Device override
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Force specific device (default: auto-detect)'
    )
    
    # Mode-specific arguments
    parser.add_argument(
        '--max-gpu-usage',
        type=float,
        default=0.95,
        help='Maximum GPU memory usage for maximum mode (default: 0.95)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['maximum', 'intensive', 'autonomous'] and args.turns:
        parser.error(f"{args.mode} mode requires --hours, not --turns")
    
    if args.mode in ['continuous', 'quick'] and args.hours and not args.turns:
        # Set default turns for these modes
        if args.mode == 'quick':
            args.turns = 20
        else:
            args.turns = int(args.hours * 20)  # Estimate turns from hours
    
    if args.memory_threshold < 0.1 or args.memory_threshold > 1.0:
        parser.error("Memory threshold must be between 0.1 and 1.0")
    
    # Override device if specified
    if args.device != 'auto':
        # This would be handled in the trainer initialization
        pass
    
    # Set memory threshold for maximum mode
    if args.mode == 'maximum':
        args.memory_threshold = args.max_gpu_usage
    
    print("🚀 Unified Document Training System")
    print("=" * 50)
      # Check for documents
    docs_path = Path(DOCUMENTS_FOLDER)
    if not docs_path.exists() or not any(docs_path.glob("*.pdf")) and not any(docs_path.glob("*.txt")):
        print(f"⚠️ No PDF or TXT documents found in {DOCUMENTS_FOLDER}/")
        print("   Place your PDF or TXT documents in the documents folder before training.")
        return 1
    
    try:
        # Initialize and run training
        trainer = UnifiedTrainer(args)
        trainer.run_training()
        return 0
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
