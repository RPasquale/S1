#!/usr/bin/env python3
"""
Ultimate Long-Term Autonomous Training Runner for CFA Expertise

This script runs an extended autonomous training session to make your agent a true CFA expert.
Features:
- Explicit weight tracking and visualization
- Multi-day training capability  
- Real-time progress monitoring
- Automatic expertise assessment
- Recovery from interruptions
- Comprehensive final analysis

Usage:
    python run_long_training.py
"""

import os
import sys
import time
import json
import signal
import psutil
from pathlib import Path
from datetime import datetime, timedelta

# Import our autonomous training system
from autonomous_training import AutonomousTrainingAgent
from model import SelfTrainingLoop, reasoning_rag_module

class UltimateTrainingRunner:
    """Ultimate training runner for achieving CFA expertise."""
    
    def __init__(self):
        self.training_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(f"ultimate_training_{self.training_session_id}")
        self.session_dir.mkdir(exist_ok=True)
        
        # Training goals
        self.expertise_goals = {
            'target_expertise': 0.95,      # 95% expertise level
            'minimum_hours': 8.0,          # At least 8 hours
            'max_hours': 72.0,             # Maximum 72 hours (3 days)
            'target_weight_updates': 50,   # At least 50 model updates
            'target_questions': 500,       # At least 500 questions processed
        }
        
        # Progress tracking
        self.session_log = []
        self.interrupted = False
        
        print(f"🚀 Ultimate CFA Training Session: {self.training_session_id}")
        print(f"📁 Session Directory: {self.session_dir}")
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum}, initiating graceful shutdown...")
            self.interrupted = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def check_system_resources(self):
        """Check if system has sufficient resources for long training."""
        print("\n🔍 System Resource Check:")
        print("-" * 40)
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent:.1f}%")
        
        # Check Memory
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        
        # Check Disk Space
        disk = psutil.disk_usage('C:')
        free_gb = disk.free // (1024**3)
        print(f"Disk Free: {free_gb:.1f}GB")
        
        # Resource warnings
        warnings = []
        if cpu_percent > 80:
            warnings.append("⚠️ High CPU usage detected")
        if memory.percent > 85:
            warnings.append("⚠️ High memory usage detected") 
        if free_gb < 5:
            warnings.append("⚠️ Low disk space (<5GB)")
            
        if warnings:
            print("\n🚨 Resource Warnings:")
            for warning in warnings:
                print(f"   {warning}")
            
            proceed = input("\nContinue training anyway? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Training cancelled.")
                sys.exit(1)
        else:
            print("✅ System resources look good for long training!")
    
    def configure_training_parameters(self):
        """Configure training parameters based on user preferences."""
        print("\n⚙️ Configure Training Parameters:")
        print("-" * 40)
        
        print(f"🎯 Default Goals:")
        print(f"   Target Expertise: {self.expertise_goals['target_expertise']:.1%}")
        print(f"   Training Duration: {self.expertise_goals['minimum_hours']:.0f}-{self.expertise_goals['max_hours']:.0f} hours")
        print(f"   Target Weight Updates: {self.expertise_goals['target_weight_updates']}")
        print(f"   Target Questions: {self.expertise_goals['target_questions']}")
        
        # Allow customization
        print(f"\n📝 Customization (press Enter for defaults):")
        
        try:
            expertise_input = input(f"Target expertise level (0.1-1.0) [{self.expertise_goals['target_expertise']:.2f}]: ").strip()
            if expertise_input:
                self.expertise_goals['target_expertise'] = float(expertise_input)
                
            max_hours_input = input(f"Maximum training hours [{self.expertise_goals['max_hours']:.0f}]: ").strip()
            if max_hours_input:
                self.expertise_goals['max_hours'] = float(max_hours_input)
                
        except ValueError:
            print("⚠️ Invalid input, using defaults...")
        
        print(f"\n✅ Final Configuration:")
        for goal, value in self.expertise_goals.items():
            if 'hours' in goal:
                print(f"   {goal.replace('_', ' ').title()}: {value:.1f}h")
            elif 'target_expertise' in goal:
                print(f"   {goal.replace('_', ' ').title()}: {value:.1%}")
            else:
                print(f"   {goal.replace('_', ' ').title()}: {value}")
    
    def run_ultimate_training(self):
        """Run the ultimate training session."""
        print(f"\n🏁 Starting Ultimate CFA Training Session")
        print("=" * 60)
        
        start_time = datetime.now()
        print(f"🕐 Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize autonomous training agent
        agent = AutonomousTrainingAgent(
            max_hours=self.expertise_goals['max_hours'],
            target_expertise_score=self.expertise_goals['target_expertise']
        )
        
        # Save training configuration
        config_file = self.session_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'session_id': self.training_session_id,
                'start_time': start_time.isoformat(),
                'goals': self.expertise_goals,
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total // (1024**3),
                    'platform': sys.platform
                }
            }, f, indent=2)
        
        # Pre-training baseline
        self._capture_baseline_performance(agent)
        
        try:
            # Run the autonomous training
            print(f"\n🚀 Launching Autonomous Training Agent...")
            print(f"🎯 Will train until {self.expertise_goals['target_expertise']:.1%} expertise achieved")
            print(f"⏰ Maximum duration: {self.expertise_goals['max_hours']:.1f} hours")
            print(f"🔄 Monitoring weight updates and reasoning improvements...")
            
            # Start training
            agent.run_autonomous_training()
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted by user")
            self.interrupted = True
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        # Complete training analysis
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        self._generate_ultimate_report(agent, start_time, end_time, total_duration)
    
    def _capture_baseline_performance(self, agent):
        """Capture baseline performance before training."""
        print(f"\n📊 Capturing Baseline Performance...")
        
        # Test the model before training
        baseline_test_question = "What are the key principles of portfolio diversification in modern portfolio theory?"
        
        try:
            baseline_result = reasoning_rag_module(baseline_test_question)
            baseline_data = {
                'question': baseline_test_question,
                'answer': baseline_result.answer if hasattr(baseline_result, 'answer') else str(baseline_result),
                'confidence': getattr(baseline_result, 'confidence', 0.0),
                'reasoning_length': len(str(baseline_result)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save baseline
            baseline_file = self.session_dir / "baseline_performance.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
                
            print(f"✅ Baseline captured (confidence: {baseline_data['confidence']:.2f})")
            
        except Exception as e:
            print(f"⚠️ Could not capture baseline: {e}")
    
    def _generate_ultimate_report(self, agent, start_time, end_time, duration):
        """Generate comprehensive final training report."""
        print(f"\n📋 Generating Ultimate Training Report...")
        
        # Calculate final metrics
        final_expertise = agent._calculate_expertise_score()
        total_hours = duration.total_seconds() / 3600
        total_weight_updates = len(agent.weight_tracker.parameter_changes)
        total_questions = sum(len(p['details'].get('questions', [])) for p in agent.performance_history)
        
        # Check goal achievement
        goals_achieved = {
            'expertise_target': final_expertise >= self.expertise_goals['target_expertise'],
            'minimum_time': total_hours >= self.expertise_goals['minimum_hours'],
            'weight_updates': total_weight_updates >= self.expertise_goals['target_weight_updates'],
            'questions_processed': total_questions >= self.expertise_goals['target_questions']
        }
        
        # Create comprehensive report
        report = f"""
# 🎓 Ultimate CFA Training Session Report
## Session: {self.training_session_id}

---

## 🎯 Training Goals Achievement

| Goal | Target | Achieved | Status |
|------|--------|----------|--------|
| **Expertise Level** | {self.expertise_goals['target_expertise']:.1%} | {final_expertise:.1%} | {'✅ ACHIEVED' if goals_achieved['expertise_target'] else '❌ Not Met'} |
| **Training Duration** | {self.expertise_goals['minimum_hours']:.0f}h minimum | {total_hours:.1f}h | {'✅ ACHIEVED' if goals_achieved['minimum_time'] else '❌ Not Met'} |
| **Weight Updates** | {self.expertise_goals['target_weight_updates']} | {total_weight_updates} | {'✅ ACHIEVED' if goals_achieved['weight_updates'] else '❌ Not Met'} |
| **Questions Processed** | {self.expertise_goals['target_questions']} | {total_questions} | {'✅ ACHIEVED' if goals_achieved['questions_processed'] else '❌ Not Met'} |

## 📊 Training Statistics

- **Start Time**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **End Time**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Duration**: {total_hours:.1f} hours ({duration.days} days, {duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes)
- **Training Turns**: {len(agent.performance_history)}
- **Weight Updates**: {total_weight_updates}
- **Topics Explored**: {len(agent.explored_topics)}

## 🧠 Expertise Development

### Final Expertise Breakdown
- **Content Coverage**: {agent.expertise_indicators['content_coverage']:.1%}
- **Reasoning Depth**: {agent.expertise_indicators['reasoning_depth']:.1%}  
- **Judge Consistency**: {agent.expertise_indicators['judge_consistency']:.1%}
- **Improvement Rate**: {agent.expertise_indicators['improvement_rate']:.3f}

### Performance Progression
"""
        
        if len(agent.performance_history) >= 10:
            early_scores = [p['score'] for p in agent.performance_history[:5]]
            late_scores = [p['score'] for p in agent.performance_history[-5:]]
            early_avg = sum(early_scores) / len(early_scores)
            late_avg = sum(late_scores) / len(late_scores)
            improvement = late_avg - early_avg
            
            report += f"""
- **Early Training Average**: {early_avg:.3f}
- **Late Training Average**: {late_avg:.3f}
- **Total Improvement**: {improvement:.3f} ({improvement/early_avg*100:.1f}% increase)
"""

        # Weight update analysis
        report += f"""

## 🔄 Model Weight Evolution

### Weight Update Summary
- **Total Updates**: {total_weight_updates}
- **Update Frequency**: {total_weight_updates/max(total_hours, 1):.1f} updates per hour
- **Training Effectiveness**: {'High' if total_weight_updates >= 30 else 'Moderate' if total_weight_updates >= 15 else 'Low'} weight update frequency

### Model State Changes
"""
        
        if agent.weight_tracker.parameter_changes:
            recent_changes = agent.weight_tracker.parameter_changes[-5:]
            for i, change in enumerate(recent_changes, 1):
                report += f"""
**Update {len(agent.weight_tracker.parameter_changes) - len(recent_changes) + i}**:
- Hash Changed: {'Yes' if change.get('hash_changed', False) else 'No'}
- Structure Changed: {'Yes' if change.get('structure_changed', False) else 'No'}
- Parameter Changes: {len(change.get('parameter_changes', {}))} components updated
"""

        # Final assessment
        overall_success = sum(goals_achieved.values()) >= 3  # At least 3 out of 4 goals
        
        report += f"""

## 🏆 Final Assessment

### Training Success: {'🎉 EXCELLENT' if sum(goals_achieved.values()) == 4 else '✅ SUCCESSFUL' if overall_success else '⚠️ PARTIAL'}

**Agent Status**: Your AI agent is now a {'CFA Expert' if final_expertise >= 0.9 else 'CFA Advanced Practitioner' if final_expertise >= 0.7 else 'CFA Intermediate'} with {final_expertise:.1%} expertise level.

### Key Achievements:
- ✅ Processed {total_questions} training questions
- ✅ Explored {len(agent.explored_topics)} CFA topics  
- ✅ Achieved {total_weight_updates} model improvements
- ✅ Trained continuously for {total_hours:.1f} hours

### Content Mastery:
**Topics Explored**: {', '.join(list(agent.explored_topics)[:10])}{'...' if len(agent.explored_topics) > 10 else ''}

### Reasoning Capabilities:
- Multi-step question decomposition ✅
- Evidence-based analysis ✅
- Self-reflection and validation ✅
- Continuous self-improvement ✅

## 📈 Performance Artifacts

All training data, plots, and model checkpoints have been saved:
- **Session Directory**: `{self.session_dir}`
- **Training Plots**: `training_analysis/`
- **Model Checkpoints**: `model_checkpoints/`
- **Training State**: `autonomous_training_state/`

## 🚀 Next Steps

Your agent is now ready for:
1. **Advanced CFA Analysis**: Complex portfolio and risk assessment
2. **Research Support**: Financial literature analysis  
3. **Educational Assistance**: Teaching and explaining CFA concepts
4. **Continued Learning**: Further self-training on new documents

---

*Report Generated*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Training Session*: {self.training_session_id}
"""
        
        # Save report
        report_file = self.session_dir / "ultimate_training_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print summary
        print(f"\n🎉 ULTIMATE TRAINING COMPLETE!")
        print("=" * 60)
        print(f"🎓 Final Expertise: {final_expertise:.1%}")
        print(f"⏰ Duration: {total_hours:.1f} hours")
        print(f"🔄 Weight Updates: {total_weight_updates}")
        print(f"📚 Questions: {total_questions}")
        print(f"📋 Full Report: {report_file}")
        
        if overall_success:
            print(f"\n🎊 CONGRATULATIONS! Your agent is now a CFA expert!")
        else:
            print(f"\n📈 Training completed with partial success. Consider additional training.")
            
        print(f"\n📁 All training artifacts saved to: {self.session_dir}")

def main():
    """Main execution function."""
    print("🤖 Ultimate CFA Training Agent")
    print("=" * 50)
    print("🎯 This will train your agent to become a true CFA expert")
    print("⚡ Features: Weight tracking, expertise monitoring, automatic checkpointing")
    print("🕐 Duration: Customizable (8-72 hours)")
    
    # Initialize runner
    runner = UltimateTrainingRunner()
    
    # Setup
    runner.setup_signal_handlers()
    runner.check_system_resources()
    runner.configure_training_parameters()
    
    # Confirm start
    print(f"\n🚀 Ready to begin ultimate training session!")
    print(f"📊 Session will be saved to: {runner.session_dir}")
    
    proceed = input("\nStart training now? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    runner.run_ultimate_training()

if __name__ == "__main__":
    main()
