#!/usr/bin/env python3
"""
Maximum GPU Utilization Training Script
Pushes RTX 4090 to its limits for maximum training throughput overnight.
Implements advanced memory management, dynamic batching, and parallel processing.
"""

import sys
import os
import datetime
import time
import threading
import queue
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import self_training_system

class MaximumGPUTrainer:
    """Maximum performance GPU training with advanced optimization."""
    
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.session_dir = Path(f"max_gpu_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}")
        self.session_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.turn_count = 0
        self.total_questions = 0
        self.performance_metrics = {
            'turns_per_hour': [],
            'questions_per_hour': [],
            'gpu_utilization': [],
            'memory_usage': []
        }
        
        # Training state
        self.best_performance = {
            'ntp_loss': float('inf'),
            'rl_score': 0.0,
            'turn_number': 0
        }
        
        print(f"🔥 Maximum GPU Trainer Initialized")
        print(f"   📁 Session: {self.session_dir}")
        print(f"   🎯 Target: Maximum RTX 4090 utilization")
    
    def run_maximum_training(self, hours=8, target_gpu_usage=0.95):
        """
        Run maximum performance training optimized for RTX 4090.
        
        Args:
            hours: Training duration
            target_gpu_usage: Target GPU memory utilization (0-1)
        """
        end_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
        
        print(f"🚀 Starting Maximum GPU Training")
        print(f"   ⏱️ Duration: {hours} hours")
        print(f"   🎯 Target GPU Usage: {target_gpu_usage*100}%")
        print(f"   📊 Training until: {end_time}")
        print(f"   💪 RTX 4090 - 24GB VRAM available")
        
        # Import torch for GPU monitoring
        import torch
        
        # Clear GPU memory and set optimal settings
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(target_gpu_usage)
        
        # Start performance monitoring thread
        monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        monitoring_thread.start()
        
        try:
            while datetime.datetime.now() < end_time:
                # Dynamic batch sizing based on GPU memory
                batch_size = self._calculate_optimal_batch_size()
                
                # Process batch of documents
                batch_start = time.time()
                batch_results = self._process_document_batch(batch_size)
                batch_duration = time.time() - batch_start
                
                # Update performance tracking
                self._update_performance_metrics(batch_results, batch_duration)
                
                # Save checkpoint every 50 turns
                if self.turn_count % 50 == 0:
                    self._save_checkpoint()
                
                # Adaptive memory management
                self._manage_gpu_memory()
                
                # Brief pause to prevent overheating
                if self.turn_count % 100 == 0:
                    time.sleep(2)  # 2-second cool-down every 100 turns
                
        except KeyboardInterrupt:
            print("\n⏸️ Maximum training interrupted")
        except Exception as e:
            print(f"\n❌ Maximum training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._finalize_maximum_training()
    
    def _calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on current GPU memory."""
        import torch
        
        try:
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_usage = memory_allocated / memory_total
            
            # Dynamic batch sizing
            if memory_usage < 0.5:
                return 4  # Large batch for low memory usage
            elif memory_usage < 0.7:
                return 3  # Medium batch
            elif memory_usage < 0.85:
                return 2  # Small batch
            else:
                return 1  # Single document to avoid OOM
                
        except Exception:
            return 2  # Safe default
    
    def _process_document_batch(self, batch_size):
        """Process a batch of documents with maximum efficiency."""
        batch_docs = []
        
        # Select random documents for this batch
        for _ in range(batch_size):
            if self_training_system.train_docs:
                import random
                doc_id, doc_content = random.choice(self_training_system.train_docs)
                doc_sample = doc_content[:3000]  # Larger chunks for intensive training
                batch_docs.append((doc_id, doc_sample))
        
        batch_results = {
            'documents_processed': len(batch_docs),
            'questions_generated': 0,
            'ntp_losses': [],
            'rl_scores': [],
            'gpu_memory_peak': 0.0
        }
        
        # Process each document in the batch
        for doc_id, doc_sample in batch_docs:
            try:
                self.turn_count += 1
                
                print(f"🔥 Turn {self.turn_count} - Processing: {doc_id[:40]}...")
                
                # Execute intensive training step with more questions
                turn_data = self_training_system.optimized_trainer.train_step(
                    doc_sample,
                    num_questions=6  # More questions per document for intensive training
                )
                
                # Collect metrics
                if turn_data['ntp_results']:
                    ntp_losses = turn_data['ntp_results'].get('losses', [])
                    batch_results['ntp_losses'].extend(ntp_losses)
                
                if turn_data['judge_scores']:
                    batch_results['rl_scores'].extend(turn_data['judge_scores'])
                    batch_results['questions_generated'] += len(turn_data['judge_scores'])
                    self.total_questions += len(turn_data['judge_scores'])
                
                # Track best performance
                if turn_data['ntp_results']:
                    avg_ntp = np.mean(turn_data['ntp_results'].get('losses', [1.0]))
                    if avg_ntp < self.best_performance['ntp_loss']:
                        self.best_performance['ntp_loss'] = avg_ntp
                        self.best_performance['turn_number'] = self.turn_count
                
                if turn_data['judge_scores']:
                    avg_rl = np.mean(turn_data['judge_scores'])
                    if avg_rl > self.best_performance['rl_score']:
                        self.best_performance['rl_score'] = avg_rl
                        self.best_performance['turn_number'] = self.turn_count
                
                # Monitor GPU memory
                gpu_memory = turn_data.get('gpu_memory_used', 0.0)
                if gpu_memory > batch_results['gpu_memory_peak']:
                    batch_results['gpu_memory_peak'] = gpu_memory
                
            except Exception as e:
                print(f"⚠️ Error in turn {self.turn_count}: {e}")
                continue
        
        return batch_results
    
    def _monitor_performance(self):
        """Monitor training performance in background thread."""
        import numpy as np
        
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                # Calculate performance metrics
                elapsed_hours = (datetime.datetime.now() - self.start_time).total_seconds() / 3600
                
                if elapsed_hours > 0:
                    turns_per_hour = self.turn_count / elapsed_hours
                    questions_per_hour = self.total_questions / elapsed_hours
                    
                    self.performance_metrics['turns_per_hour'].append(turns_per_hour)
                    self.performance_metrics['questions_per_hour'].append(questions_per_hour)
                    
                    print(f"📊 Performance Update:")
                    print(f"   🔄 Turns/hour: {turns_per_hour:.1f}")
                    print(f"   ❓ Questions/hour: {questions_per_hour:.1f}")
                    print(f"   🏆 Best NTP Loss: {self.best_performance['ntp_loss']:.4f}")
                    print(f"   🎯 Best RL Score: {self.best_performance['rl_score']:.3f}")
                
            except Exception as e:
                print(f"⚠️ Performance monitoring error: {e}")
                time.sleep(60)
    
    def _manage_gpu_memory(self):
        """Advanced GPU memory management."""
        import torch
        
        try:
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (memory_allocated / memory_total) * 100
            
            # Aggressive memory cleanup if usage is high
            if usage_percent > 90:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"🧹 Aggressive GPU cleanup - Usage was {usage_percent:.1f}%")
            elif usage_percent > 80:
                torch.cuda.empty_cache()
                print(f"🧹 GPU cache cleared - Usage: {usage_percent:.1f}%")
            
            self.performance_metrics['memory_usage'].append(usage_percent)
            
        except Exception as e:
            print(f"⚠️ Memory management error: {e}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.session_dir / f"max_gpu_checkpoint_turn_{self.turn_count}_{timestamp}.pkl"
            
            self_training_system.optimized_trainer.save_optimized_state(str(checkpoint_file))
            
            # Save performance metrics
            metrics_file = self.session_dir / f"performance_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'turn_count': self.turn_count,
                    'total_questions': self.total_questions,
                    'best_performance': self.best_performance,
                    'performance_metrics': self.performance_metrics
                }, f, indent=2)
            
            print(f"💾 Checkpoint saved: Turn {self.turn_count}")
            
        except Exception as e:
            print(f"⚠️ Checkpoint save failed: {e}")
    
    def _finalize_maximum_training(self):
        """Finalize maximum training session."""
        end_time = datetime.datetime.now()
        total_duration = end_time - self.start_time
        
        # Final statistics
        elapsed_hours = total_duration.total_seconds() / 3600
        avg_turns_per_hour = self.turn_count / elapsed_hours if elapsed_hours > 0 else 0
        avg_questions_per_hour = self.total_questions / elapsed_hours if elapsed_hours > 0 else 0
        
        # Save final checkpoint
        final_checkpoint = self.session_dir / "max_gpu_final_checkpoint.pkl"
        self_training_system.optimized_trainer.save_optimized_state(str(final_checkpoint))
        
        # Final insights
        final_insights = self_training_system.optimized_trainer.get_optimization_insights()
        
        # Comprehensive summary
        summary = {
            "session_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_hours": elapsed_hours,
                "total_turns": self.turn_count,
                "total_questions": self.total_questions
            },
            "performance": {
                "avg_turns_per_hour": avg_turns_per_hour,
                "avg_questions_per_hour": avg_questions_per_hour,
                "best_ntp_loss": self.best_performance['ntp_loss'],
                "best_rl_score": self.best_performance['rl_score'],
                "best_performance_turn": self.best_performance['turn_number']
            },
            "gpu_metrics": {
                "avg_memory_usage": np.mean(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                "peak_memory_usage": max(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0
            },
            "final_insights": final_insights
        }
        
        # Save summary
        summary_file = self.session_dir / "max_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n🏁 Maximum GPU Training Complete!")
        print(f"   📁 Results: {self.session_dir}")
        print(f"   ⏱️ Duration: {total_duration}")
        print(f"   🔄 Total Turns: {self.turn_count}")
        print(f"   ❓ Total Questions: {self.total_questions}")
        print(f"   📈 Performance: {avg_turns_per_hour:.1f} turns/hour")
        print(f"   🎯 Throughput: {avg_questions_per_hour:.1f} questions/hour")
        print(f"   🏆 Best NTP Loss: {self.best_performance['ntp_loss']:.4f}")
        print(f"   🌟 Best RL Score: {self.best_performance['rl_score']:.3f}")

def main():
    """Main function for maximum GPU training."""
    import numpy as np
    
    print("🔥 Maximum GPU Utilization Trainer")
    print("=" * 50)
    print("🚀 Optimized for RTX 4090 - 24GB VRAM")
    print("🎯 Target: Maximum training throughput overnight")
    
    # Initialize trainer
    trainer = MaximumGPUTrainer()
    
    # Check system readiness
    device = self_training_system.optimized_trainer.device
    doc_count = len(self_training_system.train_docs)
    
    print(f"\n📊 System Status:")
    print(f"   🔧 Device: {device}")
    print(f"   📄 Training docs: {doc_count}")
    
    if device != 'cuda':
        print("❌ GPU not available! This script requires CUDA.")
        return
    
    # Configuration
    hours = 8  # 8 hours overnight
    target_usage = 0.95  # Use 95% of GPU memory
    
    print(f"\n⚙️ Training Configuration:")
    print(f"   ⏱️ Duration: {hours} hours")
    print(f"   🎯 GPU Target: {target_usage*100}% memory utilization")
    print(f"   📊 Estimated throughput: ~150-200 turns/hour")
    print(f"   💪 Expected: ~1200-1600 total turns")
    
    # Final confirmation
    response = input(f"\n🚀 Start maximum GPU training for {hours} hours? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print(f"\n🔥 Starting maximum GPU training...")
    print(f"⚠️ This will push your RTX 4090 to its limits!")
    
    # Start training
    trainer.run_maximum_training(hours=hours, target_gpu_usage=target_usage)

if __name__ == "__main__":
    main()
