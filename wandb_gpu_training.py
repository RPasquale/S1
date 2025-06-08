#!/usr/bin/env python3
"""
Enhanced GPU Training with WandB Real-time Monitoring
Provides beautiful graphs for loss and RL scores during training
"""

import wandb
import torch
import time
import os
import json
import pickle
from datetime import datetime, timedelta
import pynvml
import psutil
from pathlib import Path

# Import our training system
from model import OptimizedDualTrainer, self_training_system

class WandBGPUTrainer:
    def __init__(self, project_name="dual-objective-training", run_name=None):
        """Initialize WandB GPU trainer with comprehensive monitoring"""
        self.project_name = project_name
        self.run_name = run_name or f"gpu_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.gpu_available = False
            print("⚠️  GPU monitoring not available")
        
        # Initialize trainer
        self.trainer = OptimizedDualTrainer(device='auto')
        
        # Training history
        self.training_history = []
        self.start_time = None
        
    def init_wandb(self, config=None):
        """Initialize WandB with comprehensive config"""
        default_config = {
            "learning_rate_ntp": 1e-4,
            "learning_rate_rl": 5e-5,
            "device": str(self.trainer.device),
            "gpu_available": self.gpu_available,
            "questions_per_turn": 8,
            "checkpoint_frequency": 25,
            "gpu_memory_threshold": 0.85
        }
        
        if config:
            default_config.update(config)
        
        # Initialize WandB
        wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=default_config,
            tags=["gpu", "dual-objective", "ntp", "rl"]
        )
        
        print(f"🚀 WandB initialized: {wandb.run.url}")
        return wandb.run.url
    
    def get_gpu_metrics(self):
        """Get comprehensive GPU metrics"""
        if not self.gpu_available:
            return {}
        
        try:
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used_gb = mem_info.used / (1024**3)
            memory_total_gb = mem_info.total / (1024**3)
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to watts
            
            return {
                "gpu_utilization": util.gpu,
                "gpu_memory_utilization": util.memory,
                "gpu_memory_used_gb": memory_used_gb,
                "gpu_memory_total_gb": memory_total_gb,
                "gpu_memory_percent": memory_percent,
                "gpu_temperature": temp,
                "gpu_power_watts": power,
                "torch_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
                "torch_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
            }
        except Exception as e:
            print(f"⚠️  GPU metrics error: {e}")
            return {}
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "cpu_percent": cpu_percent,
            "system_memory_percent": memory.percent,
            "system_memory_used_gb": memory.used / (1024**3),
            "system_memory_available_gb": memory.available / (1024**3)
        }
    
    def log_training_step(self, turn, ntp_loss, rl_score, elapsed_time, doc_name=""):
        """Log comprehensive training metrics to WandB"""
        # Get metrics
        gpu_metrics = self.get_gpu_metrics()
        system_metrics = self.get_system_metrics()
        
        # Calculate rates
        turns_per_hour = turn / (elapsed_time / 3600) if elapsed_time > 0 else 0
        
        # Prepare log data
        log_data = {
            # Core training metrics
            "turn": turn,
            "ntp_loss": ntp_loss,
            "rl_score": rl_score,
            "ntp_improvement": self.calculate_improvement("ntp_loss", ntp_loss),
            "rl_improvement": self.calculate_improvement("rl_score", rl_score),
            
            # Performance metrics
            "elapsed_time_hours": elapsed_time / 3600,
            "turns_per_hour": turns_per_hour,
            "document": doc_name,
            
            # System metrics
            **gpu_metrics,
            **system_metrics
        }
        
        # Log to WandB
        wandb.log(log_data)
        
        # Store in history
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "turn": turn,
            "ntp_loss": ntp_loss,
            "rl_score": rl_score,
            "elapsed_time": elapsed_time,
            **gpu_metrics,
            **system_metrics
        })
        
        # Display progress
        self.display_progress(turn, ntp_loss, rl_score, elapsed_time, gpu_metrics)
    
    def calculate_improvement(self, metric_name, current_value):
        """Calculate improvement percentage from baseline"""
        if not self.training_history:
            return 0.0
        
        baseline = self.training_history[0].get(metric_name.replace("_improvement", ""))
        if baseline is None or baseline == 0:
            return 0.0
        
        if metric_name == "ntp_loss":
            # For loss, improvement is reduction
            return ((baseline - current_value) / baseline) * 100
        else:
            # For scores, improvement is increase
            return ((current_value - baseline) / baseline) * 100
    
    def display_progress(self, turn, ntp_loss, rl_score, elapsed_time, gpu_metrics):
        """Display beautiful progress information"""
        print(f"\n{'='*60}")
        print(f"🚀 TURN {turn:,} | {elapsed_time/3600:.2f}h elapsed")
        print(f"{'='*60}")
        
        # Training metrics
        print(f"📊 NTP Loss: {ntp_loss:.4f} ({self.calculate_improvement('ntp_loss', ntp_loss):+.1f}%)")
        print(f"🎯 RL Score: {rl_score:.4f} ({self.calculate_improvement('rl_score', rl_score):+.1f}%)")
        
        # GPU metrics
        if gpu_metrics:
            print(f"🔥 GPU: {gpu_metrics.get('gpu_utilization', 0):.0f}% | "
                  f"Mem: {gpu_metrics.get('gpu_memory_percent', 0):.1f}% "
                  f"({gpu_metrics.get('gpu_memory_used_gb', 0):.1f}GB) | "
                  f"Temp: {gpu_metrics.get('gpu_temperature', 0):.0f}°C")
        
        # Performance
        turns_per_hour = turn / (elapsed_time / 3600) if elapsed_time > 0 else 0
        print(f"⚡ Speed: {turns_per_hour:.1f} turns/hour")
        
        print(f"🌐 WandB: {wandb.run.url}")
    
    def save_checkpoint(self, turn, additional_data=None):
        """Save training checkpoint"""
        checkpoint_data = {
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "trainer_state": self.trainer.__dict__.copy(),
            "training_history": self.training_history[-10:],  # Last 10 entries
            "wandb_run_id": wandb.run.id,
            "wandb_url": wandb.run.url
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        checkpoint_file = f"wandb_checkpoint_turn_{turn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"💾 Checkpoint saved: {checkpoint_file}")
            
            # Log checkpoint to WandB
            wandb.log({"checkpoint_turn": turn})
            
        except Exception as e:
            print(f"⚠️  Checkpoint save error: {e}")
    
    def run_intensive_training(self, hours=20, questions_per_turn=8, checkpoint_freq=25):
        """Run intensive training with WandB monitoring"""
        print(f"\n🚀 Starting {hours}-hour intensive GPU training with WandB monitoring!")
        print(f"📊 Questions per turn: {questions_per_turn}")
        print(f"💾 Checkpoint frequency: {checkpoint_freq} turns")
        
        # Initialize WandB
        wandb_url = self.init_wandb({
            "training_duration_hours": hours,
            "questions_per_turn": questions_per_turn,
            "checkpoint_frequency": checkpoint_freq
        })
        
        print(f"🌐 Live monitoring: {wandb_url}")
        
        # Setup
        self.start_time = time.time()
        end_time = self.start_time + (hours * 3600)
        turn = 0
          # Load documents
        print("📚 Loading documents...")
        documents = self_training_system.train_docs
        print(f"📄 Loaded {len(documents)} documents")
        
        try:
            while time.time() < end_time:
                turn += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Select document
                doc_idx = (turn - 1) % len(documents)
                doc_name, doc_content = documents[doc_idx]
                
                # Train on document
                print(f"\n📖 Training on: {doc_name}")
                
                # Get baseline metrics
                baseline_insights = self.trainer.get_optimization_insights()
                baseline_ntp = baseline_insights.get('avg_ntp_loss', 0)
                baseline_rl = baseline_insights.get('avg_rl_score', 0)
                
                # Run training
                result = self.trainer.train_on_document_optimized(
                    doc_content, 
                    num_questions=questions_per_turn,
                    chunk_size=512
                )
                
                # Get post-training metrics
                post_insights = self.trainer.get_optimization_insights()
                final_ntp = post_insights.get('avg_ntp_loss', baseline_ntp)
                final_rl = post_insights.get('avg_rl_score', baseline_rl)
                
                # Log to WandB
                self.log_training_step(
                    turn=turn,
                    ntp_loss=final_ntp,
                    rl_score=final_rl,
                    elapsed_time=elapsed_time,
                    doc_name=doc_name
                )
                
                # Checkpoint
                if turn % checkpoint_freq == 0:
                    self.save_checkpoint(turn, {"document": doc_name})
                
                # GPU memory management
                if self.gpu_available and turn % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Check remaining time
                remaining_time = end_time - current_time
                if remaining_time <= 0:
                    break
                
                print(f"⏰ Remaining: {remaining_time/3600:.2f} hours")
        
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
        
        finally:
            # Final checkpoint
            final_time = time.time()
            total_elapsed = final_time - self.start_time
            
            print(f"\n🏁 Training completed!")
            print(f"⏱️  Total time: {total_elapsed/3600:.2f} hours")
            print(f"🔄 Total turns: {turn:,}")
            print(f"📊 WandB Dashboard: {wandb.run.url}")
            
            # Final checkpoint
            self.save_checkpoint(turn, {
                "training_completed": True,
                "total_time_hours": total_elapsed/3600,
                "final_insights": self.trainer.get_optimization_insights()
            })
              # Save full history
            history_file = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            print(f"📝 History saved: {history_file}")
            
            # Safely finish WandB with error handling
            try:
                print("🔄 Finalizing WandB logging...")
                wandb.finish(quiet=True)
                print("✅ WandB training session completed")
            except Exception as wandb_error:
                print(f"⚠️  WandB finish error (training data already saved): {wandb_error}")
                print("📋 Note: Training completed successfully, only final cleanup failed")

def main():
    """Main training function"""
    # Create trainer
    trainer = WandBGPUTrainer(
        project_name="CFA-Dual-Training",
        run_name=f"intensive_gpu_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    
    # Run intensive training
    trainer.run_intensive_training(
        hours=20,              # 20-hour training
        questions_per_turn=8,  # High RL density
        checkpoint_freq=25     # Checkpoint every 25 turns
    )

if __name__ == "__main__":
    main()
