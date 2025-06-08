#!/usr/bin/env python3
"""
WandB Integration for Real-time Training Monitoring
Watch your NTP Loss, RL Scores, GPU usage, and more in beautiful graphs!
"""

import wandb
import time
import datetime
import numpy as np
import torch
import psutil
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the components we need
from model import self_training_system

class WandBTrainingMonitor:
    """Advanced WandB monitoring for dual-objective training"""
    
    def __init__(self, project_name="cfa-dual-training", run_name=None):
        self.project_name = project_name
        
        if run_name is None:
            run_name = f"overnight_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize WandB
        wandb.init(
            project=self.project_name,
            name=run_name,
            config={
                "training_type": "dual_objective",
                "ntp_lr": self_training_system.optimized_trainer.ntp_lr,
                "rl_lr": self_training_system.optimized_trainer.rl_lr,
                "device": self_training_system.optimized_trainer.device,
                "total_documents": len(self_training_system.train_docs),
                "model_type": "CFA_Expert",
                "optimization": "Adam"
            },
            tags=["overnight", "dual-training", "cfa", "gpu-accelerated"]
        )
        
        # Setup custom metrics
        wandb.define_metric("training/step")
        wandb.define_metric("ntp/*", step_metric="training/step")
        wandb.define_metric("rl/*", step_metric="training/step")
        wandb.define_metric("system/*", step_metric="training/step")
        wandb.define_metric("gpu/*", step_metric="training/step")
        
        print(f"🎯 WandB monitoring initialized: {wandb.run.url}")
    
    def log_training_step(self, step_data):
        """Log a complete training step to WandB"""
        step = step_data.get('step', 0)
        
        # NTP Metrics
        ntp_results = step_data.get('ntp_results', {})
        if ntp_results:
            ntp_loss = np.mean(ntp_results.get('losses', [1.0]))
            ntp_perplexity = np.mean(ntp_results.get('perplexities', [100.0]))
            ntp_accuracy = np.mean(ntp_results.get('accuracies', [0.0]))
            
            wandb.log({
                "training/step": step,
                "ntp/loss": ntp_loss,
                "ntp/perplexity": ntp_perplexity,
                "ntp/accuracy": ntp_accuracy,
                "ntp/best_loss": self_training_system.optimized_trainer.best_ntp_loss,
            })
        
        # RL Metrics
        rl_scores = step_data.get('judge_scores', [])
        if rl_scores:
            rl_score = np.mean(rl_scores)
            rl_std = np.std(rl_scores)
            
            wandb.log({
                "rl/average_score": rl_score,
                "rl/score_std": rl_std,
                "rl/best_score": self_training_system.optimized_trainer.best_rl_score,
                "rl/num_questions": len(rl_scores),
            })
        
        # Overall Performance
        overall_perf = step_data.get('overall_performance', 0.0)
        wandb.log({
            "training/overall_performance": overall_perf,
            "training/questions_processed": len(rl_scores) if rl_scores else 0,
        })
        
        # Learning Rates
        insights = self_training_system.optimized_trainer.optimization_insights
        wandb.log({
            "ntp/learning_rate": insights.get('learning_rate_ntp', 0),
            "rl/learning_rate": insights.get('learning_rate_rl', 0),
            "training/convergence_status": insights.get('convergence_status', 'unknown'),
        })
        
        # GPU Metrics
        self._log_gpu_metrics(step)
        
        # System Metrics
        self._log_system_metrics(step)
    
    def _log_gpu_metrics(self, step):
        """Log GPU usage metrics"""
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)   # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                memory_percent = (memory_allocated / memory_total) * 100
                
                # GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    memory_util = util.memory
                except ImportError:
                    gpu_util = 0
                    memory_util = 0
                
                wandb.log({
                    "gpu/memory_allocated_gb": memory_allocated,
                    "gpu/memory_reserved_gb": memory_reserved,
                    "gpu/memory_total_gb": memory_total,
                    "gpu/memory_percent": memory_percent,
                    "gpu/utilization_percent": gpu_util,
                    "gpu/memory_utilization_percent": memory_util,
                })
                
            except Exception as e:
                print(f"⚠️ GPU monitoring error: {e}")
    
    def _log_system_metrics(self, step):
        """Log system metrics"""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            wandb.log({
                "system/cpu_percent": cpu_percent,
                "system/memory_percent": memory.percent,
                "system/memory_used_gb": memory.used / (1024**3),
                "system/memory_total_gb": memory.total / (1024**3),
            })
            
        except Exception as e:
            print(f"⚠️ System monitoring error: {e}")
    
    def log_checkpoint(self, checkpoint_path, performance_metrics):
        """Log checkpoint creation"""
        wandb.log({
            "training/checkpoint_saved": 1,
            "training/checkpoint_path": str(checkpoint_path),
            "training/checkpoint_ntp_loss": performance_metrics.get('ntp_loss', 0),
            "training/checkpoint_rl_score": performance_metrics.get('rl_score', 0),
        })
        
        # Save checkpoint as artifact
        try:
            artifact = wandb.Artifact(
                name=f"checkpoint_step_{performance_metrics.get('step', 0)}",
                type="model_checkpoint",
                description=f"Training checkpoint at step {performance_metrics.get('step', 0)}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"⚠️ Artifact upload error: {e}")
    
    def log_training_summary(self, final_results):
        """Log final training summary"""
        wandb.log({
            "final/total_turns": final_results.get('total_turns', 0),
            "final/best_ntp_loss": final_results.get('best_ntp_loss', float('inf')),
            "final/best_rl_score": final_results.get('best_rl_score', 0.0),
            "final/duration_hours": final_results.get('duration_seconds', 0) / 3600,
        })
        
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Total Training Turns", final_results.get('total_turns', 0)],
                ["Best NTP Loss", f"{final_results.get('best_ntp_loss', 0):.4f}"],
                ["Best RL Score", f"{final_results.get('best_rl_score', 0):.3f}"],
                ["Training Duration (hours)", f"{final_results.get('duration_seconds', 0) / 3600:.1f}"],
                ["Device", self_training_system.optimized_trainer.device],
            ]
        )
        wandb.log({"training_summary": summary_table})
      def finish(self):
        """Finish WandB run with error handling"""
        try:
            print("🔄 Finalizing WandB monitoring...")
            wandb.finish(quiet=True)
            print("✅ WandB monitoring finished!")
        except Exception as wandb_error:
            print(f"⚠️  WandB finish error (data already saved): {wandb_error}")
            print("📋 Note: Monitoring completed successfully, only final cleanup failed")

# Enhanced training functions with WandB integration
def run_intensive_gpu_training_with_wandb(hours=20, questions_per_turn=8, memory_threshold=0.85):
    """
    Run intensive GPU training with real-time WandB monitoring
    """
    print(f"🚀 Starting Intensive GPU Training with WandB Monitoring")
    print(f"   - Duration: {hours} hours")
    print(f"   - Questions per turn: {questions_per_turn}")
    print(f"   - Memory threshold: {memory_threshold * 100}%")
    
    # Initialize WandB monitor
    monitor = WandBTrainingMonitor(
        project_name="cfa-expert-training",
        run_name=f"overnight_intensive_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Training parameters
    end_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
    turn_count = 0
    best_performance = {'ntp_loss': float('inf'), 'rl_score': 0.0}
    
    print(f"🎯 Training until: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Monitor live at: {wandb.run.url}")
    
    try:
        while datetime.datetime.now() < end_time:
            turn_count += 1
            
            # Monitor GPU memory and adjust batch size
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory
                
                if memory_used > memory_threshold:
                    torch.cuda.empty_cache()
                    print(f"🧹 GPU memory cleared: {memory_used*100:.1f}% usage")
            
            # Select random document
            if not self_training_system.train_docs:
                print("⚠️ No training documents available")
                break
            
            import random
            doc_id, doc_content = random.choice(self_training_system.train_docs)
            doc_sample = doc_content[:2500]  # Larger sample for intensive training
            
            print(f"\n🔄 Intensive Turn {turn_count} - Document: {doc_id[:40]}...")
            
            try:
                # Execute intensive training step
                turn_data = self_training_system.optimized_trainer.train_step(
                    doc_sample, 
                    num_questions=questions_per_turn
                )
                
                # Log to WandB
                monitor.log_training_step(turn_data)
                
                # Track performance
                if turn_data['ntp_results']:
                    current_ntp_loss = np.mean(turn_data['ntp_results'].get('losses', [1.0]))
                    if current_ntp_loss < best_performance['ntp_loss']:
                        best_performance['ntp_loss'] = current_ntp_loss
                        print(f"🎉 New best NTP loss: {current_ntp_loss:.4f}")
                
                if turn_data['judge_scores']:
                    current_rl_score = np.mean(turn_data['judge_scores'])
                    if current_rl_score > best_performance['rl_score']:
                        best_performance['rl_score'] = current_rl_score
                        print(f"🎉 New best RL score: {current_rl_score:.3f}")
                
                # Show progress every 10 turns
                if turn_count % 10 == 0:
                    remaining_time = end_time - datetime.datetime.now()
                    hours_left = remaining_time.total_seconds() / 3600
                    print(f"⏱️ Turn {turn_count} - {hours_left:.1f}h remaining")
                    print(self_training_system.optimized_trainer.get_optimization_insights())
                
                # Save checkpoint every 25 turns
                if turn_count % 25 == 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_file = f"intensive_gpu_checkpoint_turn_{turn_count}_{timestamp}.pkl"
                    self_training_system.optimized_trainer.save_optimized_state(checkpoint_file)
                    
                    # Log checkpoint to WandB
                    monitor.log_checkpoint(checkpoint_file, {
                        'step': turn_count,
                        'ntp_loss': best_performance['ntp_loss'],
                        'rl_score': best_performance['rl_score']
                    })
                    print(f"💾 Intensive checkpoint saved: {checkpoint_file}")
                
            except Exception as e:
                print(f"⚠️ Turn {turn_count} failed: {e}")
                continue
    
    except KeyboardInterrupt:
        print(f"\n⏸️ Intensive training interrupted at turn {turn_count}")
    
    # Final results
    final_results = {
        'total_turns': turn_count,
        'best_ntp_loss': best_performance['ntp_loss'],
        'best_rl_score': best_performance['rl_score'],
        'duration_seconds': hours * 3600
    }
    
    # Log final summary to WandB
    monitor.log_training_summary(final_results)
    
    print(f"\n🏁 Intensive GPU Training Completed")
    print(f"   - Total turns: {turn_count}")
    print(f"   - Best NTP Loss: {best_performance['ntp_loss']:.4f}")
    print(f"   - Best RL Score: {best_performance['rl_score']:.3f}")
    print(f"📊 Full results at: {wandb.run.url}")
    
    monitor.finish()
    return final_results

if __name__ == "__main__":
    print("🎯 WandB Training Monitor - Starting Intensive Training")
    
    # Install required packages if needed
    try:
        import wandb
        import pynvml
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install wandb pynvml")
        import wandb
    
    # Start intensive training with WandB monitoring
    run_intensive_gpu_training_with_wandb(hours=20, questions_per_turn=8, memory_threshold=0.85)
