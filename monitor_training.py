#!/usr/bin/env python3
"""
Real-Time Training Monitor

This script monitors ongoing training sessions and shows live weight updates,
performance trends, and expertise progression.

Usage:
    python monitor_training.py
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class TrainingMonitor:
    """Real-time training monitor."""
    
    def __init__(self):
        self.monitoring = True
        self.update_interval = 10  # seconds
        self.last_checkpoint_count = 0
        self.last_weight_update_count = 0
        
    def find_training_sessions(self):
        """Find active training sessions."""
        sessions = []
        
        # Look for ultimate training sessions
        for path in Path('.').glob('ultimate_training_*'):
            if path.is_dir():
                config_file = path / 'training_config.json'
                if config_file.exists():
                    sessions.append({
                        'type': 'ultimate',
                        'path': path,
                        'config_file': config_file
                    })
        
        # Look for autonomous training state
        auto_state_dir = Path('autonomous_training_state')
        if auto_state_dir.exists():
            sessions.append({
                'type': 'autonomous',
                'path': auto_state_dir,
                'config_file': None
            })
        
        return sessions
    
    def monitor_session(self, session):
        """Monitor a specific training session."""
        print(f"📊 Monitoring {session['type']} training session: {session['path']}")
        
        while self.monitoring:
            try:
                self.display_session_status(session)
                time.sleep(self.update_interval)
            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(5)
    
    def display_session_status(self, session):
        """Display current session status."""
        clear_screen()
        
        print("🔍 CFA Training Monitor")
        print("=" * 50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Session: {session['path']}")
        
        if session['type'] == 'ultimate':
            self.display_ultimate_session(session)
        else:
            self.display_autonomous_session(session)
        
        print(f"\n🔄 Auto-refresh every {self.update_interval}s (Ctrl+C to stop)")
    
    def display_ultimate_session(self, session):
        """Display ultimate training session status."""
        config_file = session['config_file']
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            start_time = datetime.fromisoformat(config['start_time'])
            elapsed = datetime.now() - start_time
            
            print(f"\n🎯 Ultimate Training Session")
            print("-" * 30)
            print(f"Start Time: {start_time.strftime('%H:%M:%S')}")
            print(f"Elapsed: {elapsed.total_seconds()/3600:.1f} hours")
            print(f"Target Expertise: {config['goals']['target_expertise']:.1%}")
            print(f"Max Duration: {config['goals']['max_hours']:.0f} hours")
            
        except Exception as e:
            print(f"⚠️ Could not read config: {e}")
        
        # Check for progress files
        progress_files = list(session['path'].glob('*.json'))
        print(f"\nProgress Files: {len(progress_files)}")
    
    def display_autonomous_session(self, session):
        """Display autonomous training session status."""
        state_dir = session['path']
        
        # Find latest training state
        state_files = list(state_dir.glob('training_state_turn_*.json'))
        
        if not state_files:
            print(f"\n⚠️ No training state files found")
            return
        
        # Get latest state
        latest_file = max(state_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        try:
            with open(latest_file, 'r') as f:
                state = json.load(f)
            
            print(f"\n🚀 Autonomous Training Status")
            print("-" * 30)
            
            # Basic info
            turn_count = state['turn_count']
            start_time = datetime.fromisoformat(state['start_time'])
            elapsed = datetime.now() - start_time
            
            print(f"Training Turns: {turn_count}")
            print(f"Start Time: {start_time.strftime('%H:%M:%S')}")
            print(f"Elapsed: {elapsed.total_seconds()/3600:.1f} hours")
            
            # Performance metrics
            if state['performance_history']:
                recent_scores = [p['score'] for p in state['performance_history'][-5:]]
                avg_recent = np.mean(recent_scores)
                print(f"Recent Avg Score: {avg_recent:.3f}")
                
                if len(state['performance_history']) >= 10:
                    early_scores = [p['score'] for p in state['performance_history'][:5]]
                    improvement = avg_recent - np.mean(early_scores)
                    print(f"Improvement: {improvement:+.3f}")
            
            # Expertise indicators
            expertise = state['expertise_indicators']
            print(f"\n🧠 Expertise Indicators:")
            print(f"Content Coverage: {expertise['content_coverage']:.1%}")
            print(f"Reasoning Depth: {expertise['reasoning_depth']:.1%}")
            print(f"Judge Consistency: {expertise['judge_consistency']:.1%}")
            
            # Content exploration
            topics_explored = len(state['explored_topics'])
            print(f"\n📚 Content Analysis:")
            print(f"Topics Explored: {topics_explored}")
            if state['explored_topics']:
                recent_topics = state['explored_topics'][-3:]
                print(f"Recent Topics: {', '.join(recent_topics)}")
            
            # Weight updates
            weight_updates = len(state.get('weight_history', []))
            print(f"\n🔄 Model Updates:")
            print(f"Weight Updates: {weight_updates}")
            
            # Alert for new updates
            if weight_updates > self.last_weight_update_count:
                new_updates = weight_updates - self.last_weight_update_count
                print(f"🆕 NEW: {new_updates} weight updates since last check!")
                self.last_weight_update_count = weight_updates
            
            # Progress visualization
            self.show_progress_bar(expertise)
            
        except Exception as e:
            print(f"⚠️ Could not read training state: {e}")
    
    def show_progress_bar(self, expertise):
        """Show progress bar for expertise."""
        overall_expertise = (
            expertise['content_coverage'] * 0.3 +
            expertise['reasoning_depth'] * 0.3 +
            expertise['judge_consistency'] * 0.2 +
            expertise.get('improvement_rate', 0) * 0.2
        )
        
        print(f"\n📊 Overall Expertise Progress:")
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * overall_expertise)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"[{bar}] {overall_expertise:.1%}")
        
        if overall_expertise >= 0.9:
            print("🏆 EXPERT LEVEL!")
        elif overall_expertise >= 0.7:
            print("🎯 ADVANCED LEVEL")
        elif overall_expertise >= 0.5:
            print("📈 INTERMEDIATE LEVEL")
        else:
            print("🌱 BEGINNER LEVEL")
    
    def check_model_checkpoints(self):
        """Check for new model checkpoints."""
        checkpoint_dir = Path('model_checkpoints')
        if not checkpoint_dir.exists():
            return
        
        checkpoints = list(checkpoint_dir.glob('*.pkl'))
        current_count = len(checkpoints)
        
        if current_count > self.last_checkpoint_count:
            new_checkpoints = current_count - self.last_checkpoint_count
            print(f"💾 NEW: {new_checkpoints} model checkpoints saved!")
            self.last_checkpoint_count = current_count
    
    def display_training_plots(self):
        """Display training analysis plots if available."""
        plots_dir = Path('training_analysis')
        if not plots_dir.exists():
            return
        
        plot_files = list(plots_dir.glob('*.png'))
        if plot_files:
            print(f"\n📈 Training plots available: {len(plot_files)}")
            for plot in plot_files:
                print(f"   • {plot.name}")

def main():
    """Main monitoring function."""
    print("🔍 CFA Training Monitor")
    print("=" * 30)
    
    monitor = TrainingMonitor()
    
    # Find training sessions
    sessions = monitor.find_training_sessions()
    
    if not sessions:
        print("❌ No active training sessions found")
        print("💡 Start training with: python run_long_training.py")
        return
    
    print(f"📊 Found {len(sessions)} training session(s)")
    
    if len(sessions) == 1:
        # Monitor single session
        session = sessions[0]
        print(f"🎯 Monitoring: {session['path']}")
        monitor.monitor_session(session)
    else:
        # Multiple sessions - let user choose
        print("\nAvailable sessions:")
        for i, session in enumerate(sessions, 1):
            print(f"  {i}. {session['type']} - {session['path']}")
        
        try:
            choice = int(input("\nSelect session to monitor (number): ")) - 1
            if 0 <= choice < len(sessions):
                monitor.monitor_session(sessions[choice])
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
